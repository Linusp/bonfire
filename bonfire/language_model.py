from operator import mul

import numpy as np

import torch
from torch.optim import Adam
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Module, Embedding, RNN, LSTM, GRU, Linear, Dropout, CrossEntropyLoss

from .vectorizer import Vectorizer
from .loss import FocalLoss, pullaway_loss


CUDA_AVAILABLE = torch.cuda.is_available()


def get_batch(seq_list, batch_size, max_step):
    for seq in seq_list:
        seq = seq[:-(len(seq) % batch_size)]  # 取长度是 batch_size 的倍数的部分
        arr = np.array(seq).reshape((batch_size, -1))
        for offset in range(0, arr.shape[1], max_step):
            yield arr[:, offset:offset + max_step]


class RNNModel(Module):

    RNN_TYPE_MAP = {
        'rnn': RNN,
        'lstm': LSTM,
        'gru': GRU,
    }

    def __init__(self, vocab_size, embedding_size, hidden_size,
                 num_layers, dropout=0.5, rnn_type='gru', tie_weights=False):
        super(RNNModel, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type

        self.dropout = Dropout(dropout)
        self.embedding = Embedding(vocab_size, embedding_size)
        rnn_cls = self.RNN_TYPE_MAP.get(rnn_type)
        self.rnn = rnn_cls(embedding_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout)
        self.linear = Linear(hidden_size, vocab_size)

        if tie_weights:
            if hidden_size == embedding_size:
                self.embedding.weight = self.linear.weight
            else:
                print("Warning: when using the `tie_weights`,"
                      "`embedding_size` should be equal to `hidden_size`")

        self.init_weights()

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
        self.linear.weight.data.uniform_(-0.1, 0.1)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        if self.rnn_type == 'lstm':
            return (Variable(weight.new(self.num_layers, batch_size, self.hidden_size).zero_()),
                    Variable(weight.new(self.num_layers, batch_size, self.hidden_size).zero_()))
        else:
            return Variable(weight.new(self.num_layers, batch_size, self.hidden_size).zero_())

    def forward(self, inputs, hidden=None):
        embedding = self.embedding(inputs)
        out, hidden_state = self.rnn(embedding, hx=hidden)
        decoded = self.linear(out.contiguous().view(out.size(0) * out.size(1), out.size(2)))
        final_out = decoded.view(out.size(0), out.size(1), decoded.size(1))

        return final_out, hidden_state


class LanguageModel(object):

    DEFAULT_VECTORIZER_CONFIG = {
        'preprocessor': ['halfwidth', 'lower'],
        'tokenizer': {'type': 'char_tokenizer'},
        'keep_puncts': True,
        'remove_non_ascii_cjk': False,
    }

    def __init__(self, vectorizer=None, embedding_size=50, hidden_size=50,
                 num_layers=2, dropout=0.5, rnn_type='gru'):
        # vectorizer 用来做向量化和逆向量化，构建的词典也存在里面
        self.vectorizer = vectorizer or Vectorizer.from_config(self.DEFAULT_VECTORIZER_CONFIG)

        # model_parameters 用来在稍后初始化 RNNLM 模型
        self.model_parameters = [
            None,               # vocab_size 占位
            embedding_size,
            hidden_size,
            num_layers,
            dropout,
            rnn_type
        ]
        self.model = None
        self.trained = False

    @classmethod
    def from_config(cls, config):
        """
        config example
        {
            'model': {
                'embedding_size': 50,
                'hidden_size': 50,4
                'num_layers': 2,
                'dropout': 0.5,
                'rnn_type': 'gru',
                'tie_weights': False,
            },
            'vectorizer': {
                'preprocessor': ['halfwidth', 'lower'],
                'tokenizer': {'type': 'char_tokenizer', 'vocabulary': ['北京', '上海']},
                'min_df': 1,
                'max_vocab_size': None,
                'keep_puncts': True,
                'remove_non_ascii_cjk': False,
            }
        }
        """
        vectorizer_config = config.get('vectorizer', {})
        vectorizer = Vectorizer.from_config(vectorizer_config)

        model_config = config.get('model', {})
        return cls(vectorizer=vectorizer, **model_config)

    def generate_next(self, char_id, hidden=None, topk=3):
        hidden = self.model.init_hidden(1) if hidden is None else hidden

        x = Variable(torch.from_numpy(np.array([[char_id]], dtype=np.float32))).long()
        if CUDA_AVAILABLE:
            x = x.cuda()
        out, hidden = self.model.forward(x, hidden)

        probs, char_ids = F.softmax(out.squeeze(), dim=-1).data.topk(topk)
        if CUDA_AVAILABLE:  # numpy 的操作不支持 cuda
            probs = probs.cpu()
            char_ids = char_ids.cpu()
        result = np.random.choice(char_ids.numpy(), p=(probs / probs.sum()).numpy())

        if isinstance(hidden, tuple):
            hidden = tuple([Variable(each.data) for each in hidden])
        else:
            hidden = Variable(hidden.data)

        return result, hidden

    def generate(self, text, max_len=30, topk=5):
        self.model.eval()
        inputs = self.vectorizer.transform([text])

        results = []
        hidden = None
        for char_id in inputs[0]:
            result_id, hidden = self.generate_next(char_id, hidden, topk)

        results.append(result_id)

        while len(results) < max_len:
            result_id, hidden = self.generate_next(results[-1], hidden, topk)
            results.append(result_id)

        text = ''.join(self.vectorizer.inverse_transform([results])[0])
        return text

    def train_epoch(self, inputs, targets, optimizer, criterion,
                    epoch_no=0, batch_size=64, max_step=50, max_norm=5, eval_step=10):
        hidden = self.model.init_hidden(batch_size)

        counter = 0
        x_generator = get_batch(inputs, batch_size, max_step)
        y_generator = get_batch(targets, batch_size, max_step)
        for x, y in zip(x_generator, y_generator):
            self.model.train()
            x = Variable(torch.from_numpy(np.array(x, dtype=np.float32))).long()
            y = Variable(torch.from_numpy(np.array(y, dtype=np.float32))).long()

            if CUDA_AVAILABLE:
                x = x.cuda()
                y = y.cuda()

            if isinstance(hidden, tuple):
                hidden = tuple([Variable(each.data) for each in hidden])
            else:
                hidden = Variable(hidden.data)

            self.model.zero_grad()  # 重置梯度
            output, hidden = self.model.forward(x, hidden)

            # 将 output 的维度进行转换:
            #   [batch_size, step_size, vocab_size] -> [batch_size * step_size, vocab_size]
            # y 是 1D 的就好
            step_size = x.size(1)  # batch 里序列的长度有可能不足 max_step
            cross_entropy_loss = criterion(
                output.view(batch_size * step_size, -1),
                y.view(batch_size * step_size).long()
            )
            focal_loss = FocalLoss(gamma=2)(
                output.view(batch_size * step_size, -1),
                y.view(batch_size * step_size).long()
            )
            ploss = pullaway_loss(output.view(batch_size * step_size, -1))
            loss = cross_entropy_loss + focal_loss + 0.1 * ploss

            loss.backward()
            torch.nn.utils.clip_grad_norm(self.model.parameters(), max_norm)
            optimizer.step()

            counter += 1
            if (counter % eval_step) == 0:
                print("Epoch: {}; Step: {}; Loss: {:.4f}".format(
                    epoch_no + 1, counter, loss.data[0]
                ))

                # 从 x 中随机挑选内容
                pos = np.random.randint(0, mul(*x.size()) - 2)
                length = np.random.randint(1, min(5, mul(*x.size()) - pos - 1))
                start_tokens = x.view(-1)[pos:pos + length].data.numpy()
                start_text = ''.join(self.vectorizer.inverse_transform([start_tokens])[0]).strip()
                if start_text:
                    result = self.generate(start_text, max_len=100)
                    print("[%s]: %r" % (start_text, result))

    def train(self, corpus, max_iter=100, batch_size=64, max_step=50,
              max_norm=5, lr=0.001, eval_step=10):
        """
        corpus: list of text: ['aabbbcc', 'bbbbaaaccc', 'aslkdfjkdsl']，每个 text 是一篇文章
        """
        # 构建词典
        self.vectorizer.fit(corpus)

        # 初始化模型
        self.model_parameters[0] = self.vectorizer.vocab_size
        self.model = RNNModel(*self.model_parameters)
        self.model.double()

        if CUDA_AVAILABLE:
            self.model.cuda()

        # 准备数据
        inputs = self.vectorizer.transform(corpus)

        # 取 inputs 中每个数据向右错一位作为 target，也就是预测下一个字
        # 例如：abcdefg -> abcdef:bcdefg
        targets = [row[1:] for row in inputs]
        inputs = [row[:-1] for row in inputs]

        # 准备目标函数和优化方法
        optimizer = Adam(self.model.parameters(), lr=lr)
        criterion = CrossEntropyLoss()

        for epoch_idx in range(max_iter):
            self.train_epoch(
                inputs, targets, optimizer, criterion, epoch_idx,
                batch_size, max_step, max_norm, eval_step
            )

        self.trained = True

    def __getstate__(self):
        # 存储时将模型转换 CPU 模式，兼容不同的环境
        return self.vectorizer, self.model_parameters, self.model.cpu()

    def __setstate__(self, state):
        vectorizer, model_parameters, model = state
        self.vectorizer = vectorizer
        self.model_parameters = model_parameters
        self.model = model
        if CUDA_AVAILABLE:
            self.model.cuda()
        self.trained = bool(self.model)

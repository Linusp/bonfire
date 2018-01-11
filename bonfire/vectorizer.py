from collections import Counter

from .utils import pad_sequences
from .tokenizer import Tokenizer
from .preprocess import Preprocessor
from .consts import PUNCTS_PAT, NON_ASCII_CJK_PAT


class Vectorizer(object):
    """
    >>> config = {
        "preprocessor": [
            "lower",
            "halfwidth",
        ],
        "tokenizer": {
            "type": "char_tokenizer"
        },
        "min_df": 1,
        "max_vocab_size": 20,
        "keep_puncts": False,
        "remove_non_ascii_cjk": True
    }
    >>> v = Vectorizer.from_config(config)
    >>> v.fit(["我想预订北京到上海的飞机！", "定一张去上海的机票。", "北京到上海的机票还有吗？"])
    >>> v.transform(["我想预订北京到上海的飞机！", "北京到上海的机票还有吗？"], with_padding=True)
    [[13 12 21 19  7  5  6  4 16 17 22 15  2]
     [ 7  5  6  4 16 17 15 18 20 14  9  2  0]]
    >>> v.inverse_transform(v.transform(["我想预订北京到上海的飞机！", "定一张去上海的机票。"]))
    [['我', '想', '预', '订', '北', '京', '到', '上', '海', '的', '飞', '机', '<PUNCT>'],
     ['定', '一', '张', '去', '上', '海', '的', '机', '票', '<PUNCT>', '<PAD>', '<PAD>', '<PAD>']]
    """

    PAD_TOKEN = '<PAD>'
    UNK_TOKEN = '<UNK>'
    PUNCT_TOKEN = '<PUNCT>'

    # config name
    TOKENIZER_CONFIG = 'tokenizer'
    PREPROCESSOR_CONFIG = 'preprocessor'
    VOCAB_CONFIG = 'vocab'
    MIN_DF_CONFIG = 'min_df'
    MAX_SIZE_CONFIG = 'max_vocab_size'
    KEEP_PUNCTS_CONFIG = 'keep_puncts'
    NON_ASCII_CJK_CONFIG = 'remove_non_ascii_cjk'

    def __init__(self, tokenizer=None, preprocessor=None,
                 vocab=None, min_df=1, max_vocab_size=None,
                 keep_puncts=False, remove_non_ascii_cjk=True, **kwargs):
        """
        tokenizer: Tokenizer, 用来产生 token, 可以设置为按字切分还是按词切分
        processors: list, 预处理方法列表，目前实现了 lower/halfwidth
        vocab: list, 实际的词表，每个元素是一个 token
        min_df: int, 最小文档频率，构建词表时使用
        max_vocab_size: int, 最大词表大小，不包括 <PAD>/<UNK>/<PUNCT>
        keep_puncts: bool, 是否保留标点，若不保留，则将所有标点替换为 <PUNCT>
        remove_non_ascii_cjk: 是否将中英文外的其他 token 替换为 <UNK>
        """
        self.tokenizer = tokenizer or Tokenizer.from_config({"type": "char_tokenizer"})
        self.preprocessor = preprocessor or Preprocessor([])

        self.vocab = vocab or []
        self.inv_vocab = {token: idx for idx, token in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        self.min_df = min_df
        self.max_vocab_size = max_vocab_size
        self.keep_puncts = keep_puncts
        self.remove_non_ascii_cjk = remove_non_ascii_cjk

        self.initialized = bool(self.vocab)

    @classmethod
    def from_config(cls, config):
        """从配置文件初始化一个 Vectorizer

        >>> config = {
            "preprocessor": [
                "lower",
                "halfwidth",
            ],
            "tokenizer": {
                "type": "char_tokenizer"
            },
            "min_df": 2,
            "max_vocab_size": 20,
            "keep_puncts": False,
            "remove_non_ascii_cjk": True
        }
        >>> v = Vectorizer.from_config(config)
        """
        tokenizer = Tokenizer.from_config(config.get(cls.TOKENIZER_CONFIG, {}))
        preprocessor = Preprocessor.from_config(config.get(cls.PREPROCESSOR_CONFIG, []))

        vocab = config.get(cls.VOCAB_CONFIG, [])

        min_df = config.get(cls.MIN_DF_CONFIG, 1)
        max_vocab_size = config.get(cls.MAX_SIZE_CONFIG)
        keep_puncts = config.get(cls.KEEP_PUNCTS_CONFIG, False)
        remove_non_ascii_cjk = config.get(cls.NON_ASCII_CJK_CONFIG, True)
        return cls(
            tokenizer=tokenizer, preprocessor=preprocessor,
            vocab=vocab, min_df=min_df, max_vocab_size=max_vocab_size,
            keep_puncts=keep_puncts, remove_non_ascii_cjk=remove_non_ascii_cjk
        )

    def tokenize(self, text):
        text = self.preprocessor(text)
        tokens = []
        for token in self.tokenizer(text):
            if not self.keep_puncts and PUNCTS_PAT.match(token):
                token = self.PUNCT_TOKEN
            if self.remove_non_ascii_cjk and NON_ASCII_CJK_PAT.match(token):
                token = self.UNK_TOKEN

            tokens.append(token)

        return tokens

    def fit(self, corpus):
        """从语料中构建出词典"""
        vocab_with_df = Counter()
        for text in corpus:
            tokens = set(self.tokenize(text))
            vocab_with_df.update(tokens)

        # 将一些特殊的 token 放到词表的最前面
        start_vocab = [self.PAD_TOKEN, self.UNK_TOKEN]
        if not self.keep_puncts:
            start_vocab.append(self.PUNCT_TOKEN)

        # 根据 min_df 和 max_vocab_size 来对 vocab 进行修改
        vocab = set()
        for token, df in vocab_with_df.most_common():
            if token in start_vocab:
                continue
            if df < self.min_df:
                break
            if self.max_vocab_size and len(vocab) >= self.max_vocab_size:
                break
            vocab.add(token)

        self.vocab = start_vocab + sorted(vocab - set(start_vocab))
        self.inv_vocab = {token: idx for idx, token in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        self.initialized = True

    def transform(self, corpus, with_padding=False):
        """将一批文本向量化

        corpus: list, 待向量化的文本
        with_padding: bool, 是否对结果进行 padding
        """
        if not self.initialized:
            raise ValueError("vectorizer is not initialized")

        result = []
        for text in corpus:
            tokens = self.tokenize(text)
            token_ids = []
            for token in tokens:
                if token in self.inv_vocab:
                    token_ids.append(self.inv_vocab.get(token))
                else:
                    token_ids.append(self.inv_vocab.get(self.UNK_TOKEN))

            result.append(token_ids)

        if not with_padding:
            return result

        maxlen = max(len(item) for item in result)
        pad_id = self.inv_vocab.get(self.PAD_TOKEN)
        return pad_sequences(result, maxlen, pad_id)

    def inverse_transform(self, data, remove_pad=True):
        """token ids -> tokens"""
        if not self.initialized:
            raise ValueError("vectorizer is not initialized")

        result = []
        for token_ids in data:
            cur = []
            for idx in token_ids:
                token = self.vocab[idx]
                if remove_pad and token == self.PAD_TOKEN:
                    continue
                cur.append(token)
            result.append(cur)

        return result


if __name__ == '__main__':
    # python -m bonfire.vectorizer
    import pickle
    config = {
        "preprocessor": ["halfwidth", "lower"],
        "tokenizer": {
            "type": "char_tokenizer",
            "vocabulary": ["北京", "上海"]
        },
        "min_df": 1,
        "max_vocab_size": 30,
        "keep_puncts": True,
        "remove_non_ascii_cjk": True
    }
    v = Vectorizer.from_config(config)
    v.fit(["我想预订北京到上海的飞机！", "定一张去上海的机票。", "北京到上海的机票还有吗？"])
    vecs = v.transform(["我想预订北京到上海的飞机！", "定一张去上海的机票。", "北京到上海的机票还有吗？"], with_padding=True)
    print(vecs)
    for tokens in v.inverse_transform(vecs):
        print(tokens)

    pickle.dump(v, open("a.pkl", "wb"))
    del v
    v = pickle.load(open("a.pkl", "rb"))
    vecs = v.transform(["我想预订北京到上海的飞机！", "定一张去上海的机票。", "北京到上海的机票还有吗？"], with_padding=True)
    print()
    print(vecs)
    for tokens in v.inverse_transform(vecs):
        print(tokens)

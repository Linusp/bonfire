import os
import sys
import pickle
from glob import glob
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa

import yaml
import click
from bonfire.language_model import LanguageModel


@click.group()
def main():
    pass


@main.command()
@click.option("-d", "--data-path", type=click.Path(exists=True),
              required=True, help="存放训练数据文件的目录")
@click.option("-c", "--conf-file", type=click.Path(exists=True),
              required=True, help="模型参数配置文件")
@click.option("-m", "--model-file", required=True, help="训练后的输出模型文件")
@click.option("--max-iter", type=int, default=100, help="最大训练迭代次数(100)")
@click.option("--batch-size", type=int, default=20, help="训练时每个 batch 的大小(20)")
@click.option("--max-step", type=int, default=50, help="训练时每个 batch 里的最大序列长度(50)")
@click.option("--eval-step", type=int, default=10, help="每隔多少步对模型进行评估(10)")
def train(data_path, conf_file, model_file, max_iter, batch_size, max_step, eval_step):
    """训练语言模型"""
    corpus = []
    for data_file in glob(os.path.join(data_path, '*')):
        corpus.append(open(data_file).read().strip())
    print("{} - Loaded corpus from path: {}".format(datetime.now(), data_path))

    config = yaml.load(open(conf_file))
    model = LanguageModel.from_config(config)
    print("{} - Loaded configuration from {} and init language model with it".format(
        datetime.now(), conf_file
    ))

    print("{} - Begin training".format(datetime.now()))
    model.train(corpus, max_iter, batch_size, max_step, eval_step=eval_step)
    print("{} - Finished training".format(datetime.now()))

    pickle.dump(model, open(model_file, 'wb'))
    print("{} - Save model to file {}".format(datetime.now(), model_file))


@main.command()
@click.option("-m", "--model-file", type=click.Path(exists=True),
              required=True, help="训练好的语言模型文件")
@click.option("-l", "--length", type=int, default=20, help="生成结果的长度(20)")
@click.option("-n", "--num", type=int, default=5, help="生成结果的数量(5)")
@click.argument("start-text")
def generate(model_file, length, num, start_text):
    """使用语言模型生成文本"""
    model = pickle.load(open(model_file, 'rb'))

    for idx in range(num):
        result = model.generate(start_text, max_len=length)
        print("[RESULT %d] %r" % (idx + 1, result))


if __name__ == '__main__':
    main()

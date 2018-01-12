bonfire
=======

practices of pytorch

Support Python3 or later

## Develop

Create virtualenv and install dependencies:

```shell
make venv && make deps
```

Unit testing

```shell
make test
```

## Usage

### RNN 语言模型

使用下面的命令来训练语言模型

```shell
python scripts/lm.py -c model_config.yaml -d data/ -m lm.model --max-iter 100 --eval-step 20
```

其中:

- `-c model_config.yaml` 指定的配置文件用来定义模型结构，可参考 `examples/language_model.yaml`

- `-d data/` 指定训练用的数据目录，该目录下所有文件都将被用作训练，请用一个文件来保存一篇文章

- `-m lm.model` 指定要输出的模型文件

其余的 `--max-iter` 等选项用来控制训练时候的一些设置，具体的选项如下

```
Usage: lm.py train [OPTIONS]

  训练语言模型

Options:
  -d, --data-path PATH   存放训练数据文件的目录  [required]
  -c, --conf-file PATH   模型参数配置文件  [required]
  -m, --model-file TEXT  训练后的输出模型文件  [required]
  --max-iter INTEGER     最大训练迭代次数(100)
  --batch-size INTEGER   训练时每个 batch 的大小(20)
  --max-step INTEGER     训练时每个 batch 里的最大序列长度(50)
  --eval-step INTEGER    每隔多少步对模型进行评估(10)
  --help                 Show this message and exit.
```

训练完成后可使用 `python scripts/lm.py generate -m lm.model <text>` 来生成文本，具体使用帮助如下

```
Usage: lm.py generate [OPTIONS] START_TEXT

  使用语言模型生成文本

Options:
  -m, --model-file PATH  训练好的语言模型文件  [required]
  -l, --length INTEGER   生成结果的长度(20)
  -n, --num INTEGER      生成结果的数量(5)
  --help                 Show this message and exit.
```

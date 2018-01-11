import re


class Tokenizer(object):
    """Tokenizer

    用来从文本中得到 token 列表

    Parameters
    ----------
    vocabulary: list, optional
        用来初始化 tokenizer 的词表，仅当使用 JiebaWordTokenizer 的时候需要

    Examples
    --------
    >>> from hatstall.tokenizer import Tokenizer
    >>> tokenizer = Tokenizer.from_config({"type": "char_tokenizer"})
    >>> tokenizer
    <CharTokenizer>
    >>> tokenizer("已结婚的和尚未结婚的。")
    ['已', '结', '婚', '的', '和', '尚', '未', '结', '婚', '的', '。']
    >>> tokenizer = Tokenizer.from_config({"type": "navie_word_tokenizer"})
    >>> tokenizer
    <NavieWordTokenizer>
    >>> tokenizer("已 结婚 的 和 尚未 结婚 的。")
    ['已', '结婚', '的', '和', '尚未', '结婚', '的', '。']
    >>> tokenizer = Tokenizer.from_config({"type": "jieba_word_tokenizer"})
    >>> tokenizer
    <JiebaWordTokenizer>
    >>> tokenizer("已结婚的和尚未结婚的。")
    ['已', '结婚', '的', '和', '尚未', '结婚', '的', '。']
    """

    def __init__(self, vocabulary=None):
        self.vocabulary = vocabulary or []
        pass

    def __call__(self, text):
        raise NotImplementedError

    def __repr__(self):
        return '<{}>'.format(self.__class__.__name__)

    @classmethod
    def from_config(cls, config):
        type_map = {
            subclass.name: subclass
            for subclass in cls.__subclasses__()
        }
        vocabulary = config.get('vocabulary', [])
        tokenizer_cls = type_map[config.get('type', 'char_tokenizer')]
        return tokenizer_cls(vocabulary=vocabulary)


class CharTokenizer(Tokenizer):
    """Tokenizer

    字符分割的 tokenizer
    """

    name = 'char_tokenizer'

    def __init__(self, vocabulary=None):
        super(CharTokenizer, self).__init__(vocabulary=vocabulary)
        pass

    def __call__(self, text):
        return list(text)


class BigramTokenizer(Tokenizer):
    """Tokenizer

    字符分割的 tokenizer
    """

    name = 'bigram_tokenizer'

    def __init__(self, vocabulary=None):
        super(BigramTokenizer, self).__init__(vocabulary=vocabulary)
        pass

    def __call__(self, text):
        if len(text) == 1:
            return [text]

        result = []
        for idx in range(len(text) - 1):
            result.append(text[idx:idx + 2])

        return result


class BaseWordTokenizer(Tokenizer):
    """NavieWordTokenizer

    按空格和标点来分割的 tokenizer
    """

    name = 'navie_word_tokenizer'

    def __init__(self, vocabulary=None):
        super(BaseWordTokenizer, self).__init__(vocabulary=vocabulary)
        self.tokenizer = re.compile(
            r'(?:[#\$&@.,;:!?\'`"~_\+\-\*\/\\|\\^=<>\[\]\(\)\{\}'
            r'\u2000-\u206f\u3000-\u303f\uff30-\uff4f'
            r'\uff00-\uff0f\uff1a-\uff20\uff3b-\uff40\uff5b-\uff65]|'
            r'[^\s#\$&@.,;:!?\'`"~_\+\-\*\/\\|\\^=<>\[\]\(\)\{\}'
            r'\u2000-\u206f\u3000-\u303f\uff30-\uff4f'
            r'\uff00-\uff0f\uff1a-\uff20\uff3b-\uff40\uff5b-\uff65]+'
            r')'
        )

    def __call__(self, text):
        return self.tokenizer.findall(text)


class JiebaWordTokenizer(Tokenizer):
    """JiebaWordTokenizer

    使用 jieba 分词器的 tokenizer，可设置词典
    """

    name = 'jieba_word_tokenizer'

    def __init__(self, vocabulary=None):
        super(JiebaWordTokenizer, self).__init__(vocabulary=vocabulary)
        import jieba
        self.tokenizer = jieba.Tokenizer()
        for word in self.vocabulary:
            self.tokenizer.add_word(word)

    def __call__(self, text):
        return [word.strip() for word in self.tokenizer.lcut(text) if word.strip()]

    def __getstate__(self):
        return self.vocabulary

    def __setstate__(self, state):
        import jieba

        self.vocabulary = state or []
        self.tokenizer = jieba.Tokenizer()
        for word in self.vocabulary:
            self.tokenizer.add_word(word)

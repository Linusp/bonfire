from html.entities import entitydefs

from .consts import WHITESPACE_PAT

PREPROCESSORS = {}


def register_preprocessor(name):
    def decorator(func):
        PREPROCESSORS[name] = func
        return func

    return decorator


@register_preprocessor('lower')
def lower(text):
    return text.lower()


@register_preprocessor('halfwidth')
def to_halfwidth(text):
    """将文本中的全角字符转换为半角字符"""
    res = ''
    for char in text:
        inside_code = ord(char)
        if inside_code == 0x3000:
            inside_code = 0x0020
        else:
            inside_code -= 0xfee0

        if inside_code < 0x0020 or inside_code > 0x7e:
            res += char
        else:
            res += chr(inside_code)

    return res


@register_preprocessor("remove_html_entity")
def remove_html_entities(text):
    """去除内容中包含的 html 实体
    注意，若和 remove_html_markup 一起使用，本方法应在其执行之后再使用
    """
    for entity_name, entity_unicode in entitydefs.items():
        text = text.replace('&{};'.format(entity_name), entity_unicode)

    return text


@register_preprocessor("unify_space")
def unify_space(text):
    return WHITESPACE_PAT.sub(' ', text)


@register_preprocessor("remove_duplicate_space")
def remove_duplicate_space(text):
    import re
    return re.sub(r'[\t \r]+', ' ', text)


class Preprocessor(object):
    def __init__(self, processors):
        self.processors = processors

    @classmethod
    def from_config(cls, config):
        processors = [PREPROCESSORS[processor] for processor in config]
        return cls(processors)

    def __call__(self, text):
        for processor in self.processors:
            text = processor(text)
        return text.strip()

    def __getstate__(self):
        inverse_map = {
            func: name
            for name, func in PREPROCESSORS.items()
        }
        return [inverse_map.get(func) for func in self.processors]

    def __setstate__(self, state):
        processors = [PREPROCESSORS[processor] for processor in state]
        self.processors = processors

import re


PUNCTS_PAT = re.compile(
    r'(?:[#\$&@.,;:!?\'`"~_\+\-\*\/\\|\\^=<>\[\]\(\)\{\}]|'
    r'[\u2000-\u206f]|'
    r'[\u3000-\u303f]|'
    r'[\uff30-\uff4f]|'
    r'[\uff00-\uff0f\uff1a-\uff20\uff3b-\uff40\uff5b-\uff65])+'
)
NON_ASCII_CJK_PAT = re.compile(
    r'[^ -~\u2000-\u206f\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff'
    '\uff00-\uffff\u2e80-\u2eff\u3000-\u303f\u31C0-\u31ef]+'
)
WHITESPACE_PAT = re.compile(
    r'[ \t\u00a0\u180e\u2000-\u200d\u2029\u202f\u205f\u2060\u3000\ufeff]'
)

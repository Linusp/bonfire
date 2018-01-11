#!/usr/bin/env python
# coding: utf-8

import pip
from pip.req import parse_requirements
from setuptools import setup, find_packages


PIP_VERSION = int(pip.__version__.split('.')[0])

VERSION = '0.1.0'
if PIP_VERSION >= 6:
    REQS = [str(ir.req) for ir in parse_requirements('requirements.txt', session=False)]
else:
    REQS = [str(ir.req) for ir in parse_requirements('requirements.txt')]


setup(
    name='bonfire',
    version=VERSION,
    description='',
    license='BSD',
    packages=find_packages(),
    install_requires=REQS,
    include_package_data=True,
    zip_safe=False,
)

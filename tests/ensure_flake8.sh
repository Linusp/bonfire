#!/bin/bash

PIP=""
if [ -e "$(which pip3.6)" ];then
    PIP=pip3.6
elif [ -e "$(which pip3.5)" ];then
    PIP=pip3.5
elif [ -e "$(which pip3)" ];then
    PIP=pip3
else
    PIP=pip
fi

${PIP} install --index-url https://pypi.doubanio.com/simple flake8 --quiet

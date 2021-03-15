# -*- coding: utf-8 -*-
# @Author       : AaronJny
# @LastEditTime : 2021-03-12
# @FilePath     : /LuWu/luwu/utils/file_util.py
# @Desc         :
import os


def abspath(filepath):
    if filepath:
        return os.path.abspath(os.path.expanduser(filepath))
    else:
        return ""


def mkdirs(dirpath):
    os.makedirs(dirpath, exist_ok=True)

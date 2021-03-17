# -*- coding: utf-8 -*-
# @Author       : AaronJny
# @LastEditTime : 2021-03-15
# @FilePath     : /LuWu/luwu/utils/file_util.py
# @Desc         :
import os
import time
from uuid import uuid1
from glob import glob
from luwu.utils import cmd_util
from loguru import logger


def abspath(filepath):
    if filepath:
        return os.path.abspath(os.path.expanduser(filepath))
    else:
        return ""


LUWU_TMP_DIR_ROOT = abspath("~/.luwu/tmp")


def mkdirs(dirpath):
    os.makedirs(dirpath, exist_ok=True)


def get_tmp_dir(dir_name=""):
    """在~/.luwu下创建一个临时文件夹，并返回创建的文件夹的绝对路径

    Args:
        dir_name (str, optional): 子路径。如果不给定，则自动生成一个随机串. Defaults to ''.
    """
    timestamp = str(int(time.time()))
    if not dir_name:
        dir_name = str(uuid1())
    dirpath = abspath(os.path.join(LUWU_TMP_DIR_ROOT, timestamp, dir_name))
    mkdirs(dirpath)
    logger.info(f"已创建临时文件夹 {dirpath} .")
    return dirpath


def clean_tmp_dir(days=3):
    """清理陆吾的临时文件夹，默认清理三天前的

    Args:
        days (int, optional): 要清理几天前的临时文件. Defaults to 3.
    """
    timestamp = int(time.time())
    days_timestamp = 86400 * days
    cnt = 0
    for dir_path in glob(os.path.join(LUWU_TMP_DIR_ROOT, "*")):
        dir_timestamp = int(dir_path.split("/")[-1])
        if timestamp - dir_timestamp > days_timestamp:
            cmd = f"rm -rf {abspath(dir_path)}"
            cmd_util.run_cmd(cmd)
            cnt += 1
    logger.info(f"已清理掉 {cnt} 个临时文件夹.")

# -*- coding: utf-8 -*-
# @Author       : AaronJny
# @LastEditTime : 2021-01-22
# @FilePath     : /app/luwu/backend/__init__.py
# @Desc         :
import functools
from loguru import logger
import traceback
from flask import jsonify


def status_code_wrapper(fix=True):
    """
    一个装饰器，用于自动为接口加上状态码和错误捕捉

    Args:
        fix: 是否需要对返回结果进行封装
    """

    def decorate(func):
        @functools.wraps(func)
        def wrap(*args, **kwargs):
            ret = {"code": 0, "msg": "请求成功", "data": {}}
            try:
                data = func(*args, **kwargs)
                if not fix:
                    ret = data
                elif data:
                    ret["data"] = data
            except Exception as e:
                logger.error(e)
                traceback.print_exc()
                ret["code"] = -1
                ret["msg"] = traceback.format_exc()
            return jsonify(ret)

        return wrap

    return decorate

# -*- coding: utf-8 -*-
# @Author       : AaronJny
# @LastEditTime : 2021-01-29
# @FilePath     : /LuWu/luwu/scripts/utils.py
# @Desc         :
import importlib
import re
from urllib.parse import urljoin

import requests
from loguru import logger
from luwu.backend.config import Config

HOST = f"http://localhost:{Config.PORT}/"


def read_train_projects():
    """获取项目列表"""
    url = urljoin(HOST, "/api/v1/project/list/")
    resp = requests.get(url, timeout=30)
    if resp.status_code != 200:
        raise Exception("接口请求失败！")
    data = resp.json()
    if data["code"] != 0:
        raise Exception("接口请求失败！")
    return data["data"]


def update_project_status(xid, status):
    """更新项目状态

    Args:
        xid (int): 项目编号
        status (int): 状态编码
    """
    url = urljoin(HOST, f"/api/v1/project/{xid}/status/update/")
    data = {"status": status}
    logger.debug(f"更新状态 {xid} {status}")
    resp = requests.post(url, json=data)
    logger.debug(resp.json())


def update_project_code(xid, code):
    """更新项目状态

    Args:
        xid (int): 项目编号
        code (str): 生成的调用代码
    """
    url = urljoin(HOST, f"/api/v1/project/{xid}/code/update/")
    data = {"code": code}
    logger.debug(f"更新调用代码 {xid} {code}")
    resp = requests.post(url, json=data)
    logger.debug(resp.json())


def get_project_by_id(xid):
    """使用项目编号获取项目信息

    Args:
        xid (int): 项目编号
    """
    url = urljoin(HOST, f"/api/v1/project/{xid}/")
    resp = requests.get(url, timeout=30)
    if resp.status_code != 200:
        raise Exception("接口请求失败！")
    data = resp.json()
    if data["code"] != 0:
        raise Exception("接口请求失败！")
    logger.debug(resp.json())
    return data["data"]


def delete_project_by_id(xid):
    """使用项目编号获取项目信息

    Args:
        xid (int): 项目编号
    """
    url = urljoin(HOST, f"/api/v1/project/{xid}/delete/physical/")
    resp = requests.get(url, timeout=30)
    if resp.status_code != 200:
        raise Exception("接口请求失败！")
    data = resp.json()
    if data["code"] != 0:
        raise Exception("接口请求失败！")
    logger.debug(resp.json())
    return data["data"]


def load_model_class(model_name):
    pattern = "<class '(.*)'>"
    text = re.findall(pattern, model_name)[0]
    *model_path, model_name = text.split(".")
    model_path = ".".join(model_path)
    model_module = importlib.import_module(model_path)
    return getattr(model_module, model_name)

# -*- coding: utf-8 -*-
# @Date         : 2020-12-31
# @Author       : AaronJny
# @LastEditTime : 2021-01-07
# @FilePath     : /LuWu/luwu/bases/model_options/sections.py
# @Desc         :
import attr


@attr.s
class BaseModelSection:

    name = attr.ib(default='通用配置')
    tip = attr.ib(default='')


class EngineSection(BaseModelSection):
    """
    模型生成引擎相关的配置
    """

    name = attr.ib(default='引擎')
    tip = attr.ib(default='这里的配置决定了模型的生成方法')


class DatasetSection(BaseModelSection):
    """
    数据集相关配置
    """
    name = attr.ib(default='数据集')
    tip = attr.ib(default='')

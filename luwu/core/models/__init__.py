# -*- coding: utf-8 -*-
# @Date         : 2020-12-30
# @Author       : AaronJny
# @LastEditTime : 2021-01-20
# @FilePath     : /LuWu/luwu/core/models/__init__.py
# @Desc         :
import attr


class BaseLuWuModel:
    """基础模型接口
    """

    def __init__(self, *args, **kwargs):
        self.args = list(args)
        self.kwargs = dict(kwargs)
        self._model = self.build_model(*args, **kwargs)

    def build_model(self, *args, **kwargs):
        raise NotImplemented


@attr.s
class ModelWebInputTextOptions:

    value = attr.ib(default='')


class ModelWebInputSelectOptions:

    value = attr.ib(default=[])


@attr.s
class ModelWebInputOptions:

    name_cn = attr.ib(default='')
    name_en = attr.ib(default='')
    input_type = attr.ib(default=ModelWebInputTextOptions())
    optional = attr.ib(default=False)


class ModelTree:
    """模型配置树
    """
    pass

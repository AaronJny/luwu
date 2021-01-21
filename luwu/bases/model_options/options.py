# -*- coding: utf-8 -*-
# @Date         : 2020-12-31
# @Author       : AaronJny
# @LastEditTime : 2021-01-07
# @FilePath     : /LuWu/luwu/bases/model_options/options.py
# @Desc         :
import attr


@attr.s
class BaseModelOption:

    label = attr.ib(default='')
    option_type = attr.ib(default='')
    tip = attr.ib(default='')
    data = attr.ib(default='')


@attr.s
class InputOption(BaseModelOption):

    label = attr.ib(default='文本输入项')
    option_type = attr.ib(default='input')
    data = attr.ib(defalut={
        'placeholder': '',
        'value': ''
    })


@attr.s
class RadioGroupOption(BaseModelOption):

    label = attr.ib(default='')
    option_type = attr.ib(default='radio')
    data = attr.ib(default=[])

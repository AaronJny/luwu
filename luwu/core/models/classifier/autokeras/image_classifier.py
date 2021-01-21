# -*- coding: utf-8 -*-
# @Date         : 2020-12-30
# @Author       : AaronJny
# @LastEditTime : 2021-01-07
# @FilePath     : /LuWu/luwu/core/models/classifier/autokeras/image_classifier.py
# @Desc         :
import os
from typing import Optional, Tuple, Union

import tensorflow as tf
from luwu.core.models import (BaseLuWuModel,
                              ModelWebInputOptions,
                              ModelWebInputSelectOptions,
                              ModelWebInputTextOptions)

import autokeras as ak
from autokeras.utils import types


class LuWuAutoKerasImageClassifier(BaseLuWuModel):
    """
    autokeras 中的 ImageClassifier 封装
    """

    OPTIONS = [
        ModelWebInputOptions(name_cn='训练数据路径',
                             name_en='train data path',
                             input_type=ModelWebInputTextOptions(value=''),
                             optional=False),
        ModelWebInputOptions(name_cn='单次尝试的最大Epochs数',
                             name_en='max epochs',
                             input_type=ModelWebInputTextOptions(value='20'),
                             optional=True),
        ModelWebInputOptions(name_cn='最多搜索多少个模型结构',
                             name_en='max_trials',
                             input_type=ModelWebInputTextOptions(value='10'),
                             optional=True),
        ModelWebInputOptions(name_cn='优化指标',
                             name_en='objective',
                             input_type=ModelWebInputSelectOptions(
                                 value=['val_loss', 'val_accuracy']),
                             optional=True)
    ]

    def __init__(self):
        super(LuWuAutoKerasImageClassifier, self).__init__()

    def build_model(self, *args, **kwargs):
        """
        创建模型
        """
        kwargs['overwrite'] = True
        _model = ak.ImageClassifier(*args, **kwargs)
        return _model

    def fit(self,
            x: Optional[types.DatasetType] = None,
            y: Optional[types.DatasetType] = None,
            epochs: Optional[int] = None,
            callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
            validation_split: Optional[float] = 0.2,
            validation_data: Union[
            tf.data.Dataset, Tuple[types.DatasetType, types.DatasetType], None
            ] = None,
            *args,
            **kwargs):
        """训练模型
        """
        self._model: ak.ImageClassifier = self._model
        self._model.fit(x, y, epochs, callbacks,
                        validation_split, validation_data, **kwargs)

    def export_model(self, path):
        """导出model到指定路径
        """
        model = self._model.export_model()
        if os.path.isdir(path):
            save_path = os.path.join(path, 'best_model.h5')
            model.save(save_path)
        else:
            model.save(path)

    def export_predict_code(self, path):
        """导出调用代码到指定路径下

        Args:
            path (str): 保存预测示例的路径
        """
        pass


def gen_luwu_autokeras_image_classifier(options: dict):
    kwargs = {}
    # 类别数量
    num_classes = options.get('num_classes', 0)
    if num_classes:
        kwargs['num_classes'] = num_classes
    # 数据集路径
    dataset_path = options['dateset_path']
    dataset_type = options['dataset_type']
    if dataset_type == 'dir':
        pass
    elif dataset_type == 'text':
        pass
    else:
        raise Exception(f'未指定的数据集类型 {dataset_type}')

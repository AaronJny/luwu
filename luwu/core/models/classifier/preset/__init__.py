# -*- coding: utf-8 -*-
# @Date         : 2021-01-21
# @Author       : AaronJny
# @LastEditTime : 2021-01-22
# @FilePath     : /LuWu/luwu/core/models/classifier/preset/__init__.py
# @Desc         :
from luwu.core.models.classifier.preset.pre_trained import (
    LuwuDenseNet121ImageClassifier,
    LuwuDenseNet169ImageClassifier,
    LuwuDenseNet201ImageClassifier,
    LuwuVGG16ImageClassifier,
    LuwuVGG19ImageClassifier,
    LuwuMobileNetImageClassifier,
    LuwuMobileNetV2ImageClassifier,
    LuwuInceptionResNetV2ImageClassifier,
    LuwuInceptionV3ImageClassifier,
    LuwuNASNetMobileImageClassifier,
    LuwuNASNetLargeImageClassifier,
)

__all__ = [
    "LuwuPreTrainedImageClassifier",
    "LuwuDenseNet121ImageClassifier",
    "LuwuDenseNet169ImageClassifier",
    "LuwuDenseNet201ImageClassifier",
    "LuwuVGG16ImageClassifier",
    "LuwuVGG19ImageClassifier",
    "LuwuMobileNetImageClassifier",
    "LuwuMobileNetV2ImageClassifier",
    "LuwuInceptionResNetV2ImageClassifier",
    "LuwuInceptionV3ImageClassifier",
    "LuwuNASNetMobileImageClassifier",
    "LuwuNASNetLargeImageClassifier",
]

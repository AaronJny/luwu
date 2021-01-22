# -*- coding: utf-8 -*-
# @Date         : 2021-01-21
# @Author       : AaronJny
# @LastEditTime : 2021-01-22
# @FilePath     : /LuWu/test/test_preset_image_classifier.py
# @Desc         :
from luwu.core.models.classifier.preset import LuwuMobileNetV2ImageClassifier

path = "./images2"
LuwuMobileNetV2ImageClassifier(
    origin_dataset_path=path, target_dataset_path=path, model_save_path=path, epochs=10
).run()

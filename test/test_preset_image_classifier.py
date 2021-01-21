# -*- coding: utf-8 -*-
# @Date         : 2021-01-21
# @Author       : AaronJny
# @LastEditTime : 2021-01-21
# @FilePath     : /app/test/test_preset_image_classifier.py
# @Desc         :
from luwu.core.models.classifier.preset.desnet import LuwuDenseNet121ImageClassifier

path = "./images2"
LuwuDenseNet121ImageClassifier(
    origin_dataset_path=path, target_dataset_path=path, model_save_path=path, epochs=10
).run()

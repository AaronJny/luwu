# -*- coding: utf-8 -*-
# @Author       : AaronJny
# @LastEditTime : 2021-01-22
# @FilePath     : /app/test/test_preset_image_classifier.py
# @Desc         :
from luwu.core.models.image import (
    LuwuMobileNetV2ImageClassifier as ImageClassifier,
)

path = "./images2"
ImageClassifier(
    origin_dataset_path=path, target_dataset_path=path, model_save_path=path, epochs=10
).run()

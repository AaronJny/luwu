# -*- coding: utf-8 -*-
# @Author       : AaronJny
# @LastEditTime : 2021-03-12
# @FilePath     : /LuWu/test/test_preset_image_classifier.py
# @Desc         :
from luwu.core.models.image import (
    LuwuResNet50V2ImageClassifier as ImageClassifier,
)

# path = "./images2/"
# path = "/app/images2"
ImageClassifier(
    origin_dataset_path="/Users/aaron/test_kaggle_api/chinese-click-demo-api/data",
    target_dataset_path="/Users/aaron/test_luwu",
    model_save_path="/Users/aaron/test_luwu",
    epochs=10,
).run()

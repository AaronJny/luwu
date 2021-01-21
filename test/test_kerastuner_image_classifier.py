# -*- coding: utf-8 -*-
# @Date         : 2021-01-20
# @Author       : AaronJny
# @LastEditTime : 2021-01-20
# @FilePath     : /LuWu/test/test_kerastuner_image_classifier.py
# @Desc         :
from luwu.core.models.classifier.kerastuner.image_classifier import LuWuKerasTunerImageClassifier

path = './images2'
luwu_classifier = LuWuKerasTunerImageClassifier(path, path)
luwu_classifier.run()

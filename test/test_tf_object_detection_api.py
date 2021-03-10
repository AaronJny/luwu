# -*- coding: utf-8 -*-
# @Author       : AaronJny
# @LastEditTime : 2021-03-07
# @FilePath     : /LuWu/test/test_tf_object_detection_api.py
# @Desc         :
from luwu.core.models.complex.od.models import LuWuTFModelsObjectDetector


LuWuTFModelsObjectDetector(
    origin_dataset_path="/Users/aaron/od_api/VOCdevkit/VOC2007/yd_train_clean",
    tfrecord_dataset_path="/Users/aaron/test_luwu/dataset/",
    fine_tune_model_name="SSD ResNet50 V1 FPN 640x640 (RetinaNet50)",
    model_save_path="~/test_luwu/",
    batch_size=1,
).run()

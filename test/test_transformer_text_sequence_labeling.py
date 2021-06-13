# -*- coding: utf-8 -*-
# @Author       : AaronJny
# @LastEditTime : 2021-06-13
# @FilePath     : /LuWu/test/test_transformer_text_sequence_labeling.py
# @Desc         :
import os

os.environ["TF_KERAS"] = "1"
from luwu.core.models.text_classifier.transformers import (
    TransformerTextSequenceLabeling,
)

TransformerTextSequenceLabeling(
    origin_dataset_path="/Users/aaron/dataset/labeling_data (1)/mydata/train.jsonl",
    validation_dataset_path="/Users/aaron/dataset/labeling_data (1)/mydata/dev.jsonl",
    test_dataset_path="/Users/aaron/dataset/labeling_data (1)/mydata/test.jsonl",
    epochs=1,
    batch_size=4,
).run()

# -*- coding: utf-8 -*-
# @Author       : AaronJny
# @LastEditTime : 2021-05-11
# @FilePath     : /LuWu/test/test_transformer_text_classifier.py
# @Desc         :
from luwu.core.models.text_classifier.transformers import TransformerTextClassification

TransformerTextClassification(
    origin_dataset_path="/Users/aaron/code/luwu_nlp/classification.jsonl",
    batch_size=32,
    epochs=2,
    do_sample_balance="over",
).run()

# -*- coding: utf-8 -*-
# @Author       : AaronJny
# @LastEditTime : 2021-06-13
# @FilePath     : /LuWu/luwu/core/models/text_classifier/templates/TransformerTextSequenceLabeling.py
# @Desc         :
import os

os.environ["TF_KERAS"] = "1"
import tensorflow as tf
import tensorflow.keras.backend as K
from bert4keras.tokenizers import Tokenizer
from luwu.core.models.text_classifier.transformers import (
    NamedEntityRecognizer,
    TransformerTextSequenceLabeling,
)

# 词汇表地址
dict_path = "/Users/aaron/.luwu/cache_models/chinese_L-12_H-768_A-12/vocab.txt"
# 训练好的模型保存路径
model_path = "/Users/aaron/dataset/labeling_data (1)/mydata/luwu-text-transformer-project/best_weights.h5"
# 要预测的文本
sentence = "请替换成要进行预测的文本"
# 编号->标签的映射
categories = ["A1", "A2", "A3", "A4", "P", "T"]

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)
# 加载模型
model = tf.keras.models.load_model(model_path, compile=False)

NER = NamedEntityRecognizer(
    tokenizer,
    model,
    categories,
    trans=K.eval(
        model.get_layer("CRF").trans,
    ),
    starts=[0],
    ends=[0],
)
data = NER.recognize(sentence)
for start, end, label in data:
    print(sentence[start : end + 1], label)
print(data)

# -*- coding: utf-8 -*-
# @Author       : AaronJny
# @LastEditTime : 2021-05-13
# @FilePath     : /LuWu/luwu/core/models/text_classifier/templates/TransformerTextClassifier.py
# @Desc         :
import os

os.environ["TF_KERAS"] = "1"
import tensorflow as tf
from bert4keras.snippets import to_array
from bert4keras.tokenizers import Tokenizer
from luwu.core.models.text_classifier.transformers import TransformerTextClassification

# 词汇表地址
dict_path = ""
# 训练好的模型保存路径
model_path = ""
# 要预测的文本
sentence = ""
# 编号->标签的映射
id_label_dict = {0: "类别1", 1: "类别2", 2: "类别3"}

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)
# 加载模型
model = tf.keras.models.load_model(model_path)

# 处理文本数据
token_ids, segment_ids = tokenizer.encode(sentence)
token_ids, segment_ids = to_array([token_ids], [segment_ids])

# 预测
outputs = model.predict([token_ids, segment_ids])
index = int(tf.argmax(outputs[0]))
print("当前文本类别为：{}".format(id_label_dict[index]))

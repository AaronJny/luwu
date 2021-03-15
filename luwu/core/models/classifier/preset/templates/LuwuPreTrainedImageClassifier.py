# -*- coding: utf-8 -*-
# @Date         : 2021-01-21
# @Author       : AaronJny
# @LastEditTime : 2021-03-12
# @FilePath     : /LuWu/luwu/core/models/classifier/preset/templates/LuwuPreTrainedImageClassifier.py
# @Desc         :
import tensorflow as tf


def read_image(path):
    return ""


# 定义编号->类别的映射
num_to_classes_map = {}
# 加载训练好的模型
model = tf.keras.models.load_model("")
# 加载并预处理图片
image = read_image("/path/to/image")
# 进行预测，并输出结果
outputs = model.predict(tf.reshape(image, (-1, 224, 224, 3)))
index = int(tf.argmax(outputs[0]))
print("当前图片类别为：{}".format(num_to_classes_map[index]))

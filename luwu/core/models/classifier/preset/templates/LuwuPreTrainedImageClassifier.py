# -*- coding: utf-8 -*-
# @Date         : 2021-01-21
# @Author       : AaronJny
# @LastEditTime : 2021-01-21
# @FilePath     : /app/luwu/core/models/classifier/preset/templates/LuwuPreTrainedImageClassifier.py
# @Desc         :
import tensorflow as tf


def read_image(path):
    return ""


def build_model(num_classes=10, net_name=""):
    pre_trained_net: tf.keras.Model = tf.keras.applications.DenseNet121()
    pre_trained_net.trainable = False
    # 记录densenet
    model = tf.keras.Sequential(
        [
            pre_trained_net,
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1000, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.categorical_crossentropy,
        metrics=["accuracy"],
    )
    return pre_trained_net, model


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

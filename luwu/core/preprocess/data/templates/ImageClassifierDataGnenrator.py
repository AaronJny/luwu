# -*- coding: utf-8 -*-
# @Date         : 2021-01-21
# @Author       : AaronJny
# @LastEditTime : 2021-01-21
# @FilePath     : /LuWu/luwu/core/preprocess/data/templates/ImageClassifierDataGnenrator.py
# @Desc         :
import tensorflow as tf


def read_image(image_file_path: str):
    """从指定路径读取一张图片，并进行预处理

    Args:
        image_file_path (str): 待处理读片
    """
    image = tf.io.read_file(image_file_path)
    image = tf.io.decode_jpeg(image, channels=3)
    image_size = 224
    image = tf.image.resize(image, [image_size, image_size])
    image = tf.cast(image, dtype=tf.float32) / 255.0
    return image


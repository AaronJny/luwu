# -*- coding: utf-8 -*-
# @Date         : 2021-01-21
# @Author       : AaronJny
# @LastEditTime : 2021-01-28
# @FilePath     : /app/luwu/core/preprocess/data/templates/ImageClassifierDataGnenrator.py
# @Desc         :
import tensorflow as tf


def read_image(image_file_path: str):
    """从指定路径读取一张图片，并进行预处理

    Args:
        image_file_path (str): 待处理图片
    """
    image = tf.io.read_file(image_file_path)
    image = tf.io.decode_jpeg(image, channels=3)
    # imagenet数据集均值
    image_mean = [0.485, 0.456, 0.406]
    # imagenet数据集标准差
    image_std = [0.299, 0.224, 0.225]
    # 缩放图片
    image = tf.image.resize(image, [224, 224])
    # 将图片的像素值缩放到[0,1]之间
    image = tf.cast(image, dtype=tf.float32) / 255.0
    # 归一化
    image = (image - image_mean) / image_std
    return image

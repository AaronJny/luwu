# -*- coding: utf-8 -*-
# @Date         : 2021-01-07
# @Author       : AaronJny
# @LastEditTime : 2021-04-15
# @FilePath     : /LuWu/luwu/core/preprocess/image/load.py
# @Desc         :
import imghdr
import os
from glob import glob
from typing import List, Tuple

import tensorflow as tf
from tqdm import tqdm


def write_tfrecords_to_target_path(
    data: List[Tuple[str, str]], classes_num: int, target_dataset_path: str
):
    """将给定数据制作成tfrecords格式，并写入到目标路径

    Args:
        data (List[Tuple[str,str]]): 待写入数据
        classes_num (int): 共有多少类别
        target_dataset_path (str): 目标地址
    """
    with tf.io.TFRecordWriter(target_dataset_path) as writer:
        for image, num in tqdm(data):
            with open(image, "rb") as f:
                image_data = f.read()
            feature = {
                "image": tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=[
                            image_data,
                        ]
                    )
                ),
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[num])),
                "num": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[classes_num])
                ),
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())


def is_image(file_path):
    """判断给定文件是否为图片"""
    return imghdr.what(file_path) == "jpeg"


def read_classify_dataset_from_dir(dataset_path: str, classes_num_dict=None):
    """从给定地址读取并清洗按文件夹组织的图片数据集

    Args:
        dataset_path (str): 原始数据集地址
    """
    if classes_num_dict is None:
        classes_num_dict = {}
    # image_suffixes = {"jpg", "jpeg", "png"}
    data = []
    # 创建固定的标签映射
    classes_name = []
    for sub_path in glob(os.path.join(dataset_path, "*")):
        if os.path.isdir(sub_path):
            class_name = sub_path.split("/")[-1]
            classes_name.append(class_name)
    classes_name = sorted(classes_name)
    classes_num_dict = dict(zip(classes_name, range(len(classes_name))))
    # 逐条处理图片
    for sub_path in glob(os.path.join(dataset_path, "*")):
        if os.path.isdir(sub_path):
            class_name = sub_path.split("/")[-1]
            # classes_num_dict[class_name] = len(classes_num_dict)
            for image in glob(os.path.join(sub_path, "*")):
                # if image.split(".")[-1].lower() in image_suffixes:
                if is_image(image):
                    data.append((image, classes_num_dict[class_name]))
    return data, classes_num_dict


if __name__ == "__main__":
    read_classify_dataset_from_dir(
        "/Users/aaron/code/pets_classifer/images",
        "/Users/aaron/code/pets_classifer/images/dataset",
    )

# -*- coding: utf-8 -*-
# @Date         : 2021-01-20
# @Author       : AaronJny
# @LastEditTime : 2021-04-21
# @FilePath     : /LuWu/luwu/core/preprocess/data/data_generator.py
# @Desc         :
import os

import tensorflow as tf
from jinja2 import Template
from luwu.core.preprocess.image.process import (
    extract_image_and_label_from_record,
    normalized_image,
    normalized_image_with_imagenet,
    image_random_flip_horizontal,
    image_random_flip_vertical,
    image_random_crop,
    image_random_brightness,
    image_random_hue,
)


class BaseDataGenerator(object):
    def __init__(self, data_path, batch_size=32, shuffle=False, **kwargs):
        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._steps = -1
        self.dataset = self.load_dataset()

    def load_dataset(self):
        raise NotImplementedError

    def for_fit(self):
        return self.dataset

    @property
    def steps(self):
        if self._steps < 0:
            raise Exception("请在 `load_dataset` 中统计 steps!")
        else:
            return self._steps


class ImageClassifierDataGnenrator(BaseDataGenerator):
    def __init__(
        self,
        data_path: str,
        batch_size: int = 32,
        image_size: int = 224,
        shuffle: bool = False,
        with_image_net=True,
        image_augmentation_random_flip_horizontal: bool = False,
        image_augmentation_random_flip_vertival: bool = False,
        image_augmentation_random_crop: bool = False,
        image_augmentation_random_brightness: bool = False,
        image_augmentation_random_hue: bool = False,
        **kwargs
    ):
        """
        Args:
            data_path (str): 数据集路径
            batch_size (int, optional): mini batch大小. Defaults to 32.
            image_size (int, optional): 图片尺寸. Defaults to 224.
            shuffle (bool, optional): 是否在每次训练时对数据进行混洗. Defaults to False.
            with_image_net (bool, optional): 是否使用imagenet数据集进行归一化. Defaults to True.
            image_augmentation_random_flip_horizontal (bool): 数据增强选项，是否做随机左右镜像。默认False.
            image_augmentation_random_flip_vertival (bool): 数据增强选项，是否做随机上下镜像。默认False.
            image_augmentation_random_crop (bool): 数据增强选项，是否做随机剪裁，剪裁尺寸为原来比例的0.9。默认False.
            image_augmentation_random_brightness (bool): 数据增强选项，是否做随机饱和度调节。默认False.
            image_augmentation_random_hue (bool): 数据增强选项，是否做随机色调调节。默认False.
        """
        self.image_size = image_size
        self.with_image_net = with_image_net
        self.image_augmentation_random_flip_horizontal = (
            image_augmentation_random_flip_horizontal
        )
        self.image_augmentation_random_flip_vertival = (
            image_augmentation_random_flip_vertival
        )
        self.image_augmentation_random_crop = image_augmentation_random_crop
        self.image_augmentation_random_brightness = image_augmentation_random_brightness
        self.image_augmentation_random_hue = image_augmentation_random_hue
        super(ImageClassifierDataGnenrator, self).__init__(
            data_path, batch_size=batch_size, shuffle=shuffle, **kwargs
        )

    def load_dataset(self):
        dataset = tf.data.TFRecordDataset(
            self.data_path, num_parallel_reads=tf.data.experimental.AUTOTUNE
        )
        dataset = dataset.map(
            extract_image_and_label_from_record,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        # 数据增强：左右镜像
        if self.image_augmentation_random_flip_horizontal:
            dataset = dataset.map(
                image_random_flip_horizontal,
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
        # 数据增强：上下镜像
        if self.image_augmentation_random_flip_vertival:
            dataset = dataset.map(
                image_random_flip_vertical,
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
        # 数据增强：随机剪裁
        if self.image_augmentation_random_crop:
            dataset = dataset.map(
                image_random_crop,
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
        # 数据增强：随机饱和度调节
        if self.image_augmentation_random_brightness:
            dataset = dataset.map(
                image_random_brightness,
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
        # 数据增强：随机色调调节
        if self.image_augmentation_random_hue:
            dataset = dataset.map(
                image_random_hue,
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
        dataset = dataset.map(
            lambda x, y: (x, y, self.image_size),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        if self.with_image_net:
            dataset = dataset.map(
                normalized_image_with_imagenet,
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
        else:
            dataset = dataset.map(
                normalized_image, num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
        if self.shuffle:
            dataset = dataset.shuffle(10000)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE).batch(
            self.batch_size
        )
        # 计算总步数
        cnt = 0
        for _ in dataset:
            cnt += 1
        self._steps = cnt
        dataset = dataset.repeat()
        return dataset

    def generate_preprocess_code(self):
        """生成数据处理的代码"""
        template_path = os.path.join(
            os.path.dirname(__file__), "templates/ImageClassifierDataGnenrator.txt"
        )
        with open(template_path, "r") as f:
            text = f.read()
        template = Template(text)
        return template.render(
            image_size=self.image_size, with_image_net=self.with_image_net
        )

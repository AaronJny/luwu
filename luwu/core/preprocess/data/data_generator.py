# -*- coding: utf-8 -*-
# @Date         : 2021-01-20
# @Author       : AaronJny
# @LastEditTime : 2021-03-16
# @FilePath     : /LuWu/luwu/core/preprocess/data/data_generator.py
# @Desc         :
import os

import tensorflow as tf
from jinja2 import Template
from luwu.core.preprocess.image.process import (
    extract_image_and_label_from_record,
    normalized_image,
    normalized_image_with_imagenet,
)


class BaseDataGenerator(object):
    def __init__(
        self,
        data_path,
        batch_size=32,
        image_size=224,
        shuffle=False,
        with_image_net=True,
    ):
        self.data_path = data_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.with_image_net = with_image_net
        self._steps = -1
        self.dataset = self.load_dataset()

    def load_dataset(self):
        raise NotImplementedError

    def for_fit(self):
        yield from self.dataset

    @property
    def steps(self):
        if self._steps < 0:
            raise Exception("请在 `load_dataset` 中统计 steps!")
        else:
            return self._steps


class ImageClassifierDataGnenrator(BaseDataGenerator):
    def load_dataset(self):
        dataset = tf.data.TFRecordDataset(
            self.data_path, num_parallel_reads=tf.data.experimental.AUTOTUNE
        )
        dataset = dataset.map(
            extract_image_and_label_from_record,
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
        # todo:增加图像增广相关功能
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

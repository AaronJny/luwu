# -*- coding: utf-8 -*-
# @Date         : 2021-01-20
# @Author       : AaronJny
# @LastEditTime : 2021-01-20
# @FilePath     : /LuWu/luwu/core/preprocess/data/data_generator.py
# @Desc         :
import tensorflow as tf
from luwu.core.preprocess.image.process import (
    extract_image_and_label_from_record,
    normalized_image,
    normalized_image_with_imagenet,
)


class BaseDataGenerator(object):
    def __init__(self, data_path, batch_size=32, shuffle=True):
        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._steps = -1
        self.dataset = self.load_dataset()

    def load_dataset(self):
        raise NotImplementedError

    def for_fit(self):
        yield from self.dataset

    @property
    def steps(self):
        if self._steps < 0:
            cnt = 0
            for _ in self.dataset:
                cnt += 1
            self._steps = cnt
        else:
            return self._steps


class ImageClassifierDataGnenrator(BaseDataGenerator):
    def load_dataset(self):
        dataset = tf.data.TFRecordDataset(self.data_path)
        dataset = dataset.map(extract_image_and_label_from_record)
        dataset = dataset.map(normalized_image)
        # todo:增加图像增广相关功能
        if self.shuffle:
            dataset = dataset.shuffle(10000)
        dataset = dataset.prefetch(self.batch_size).batch(self.batch_size)
        return dataset

    def generate_preprocess_code(self):
        """"""

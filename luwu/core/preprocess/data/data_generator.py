# -*- coding: utf-8 -*-
# @Date         : 2021-01-20
# @Author       : AaronJny
# @LastEditTime : 2021-06-13
# @FilePath     : /LuWu/luwu/core/preprocess/data/data_generator.py
# @Desc         :
import math
import os

os.environ["TF_KERAS"] = "1"

import numpy as np
import tensorflow as tf
from bert4keras.snippets import sequence_padding
from jinja2 import Template
from luwu.core.preprocess.image.process import (
    extract_image_and_label_from_record,
    image_random_brightness,
    image_random_crop,
    image_random_flip_horizontal,
    image_random_flip_vertical,
    image_random_hue,
    normalized_image,
    normalized_image_with_imagenet,
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


class DataGenerator(object):
    """数据生成器模版(修改自 苏剑林 大佬的 bert4keras.)"""

    def __init__(self, data, batch_size=32, buffer_size=None):
        self.data = data
        self.batch_size = batch_size
        if hasattr(self.data, "__len__"):
            self.steps = int(math.ceil(len(self.data) / self.batch_size))
        else:
            self.steps = None
        self.buffer_size = buffer_size or batch_size * 1000

    def __len__(self):
        return self.steps

    def sample(self, random=False):
        """采样函数，每个样本同时返回一个is_end标记"""
        if random:
            if self.steps is None:

                def generator():
                    caches, isfull = [], False
                    for d in self.data:
                        caches.append(d)
                        if isfull:
                            i = np.random.randint(len(caches))
                            yield caches.pop(i)
                        elif len(caches) == self.buffer_size:
                            isfull = True
                    while caches:
                        i = np.random.randint(len(caches))
                        yield caches.pop(i)

            else:

                def generator():
                    for i in np.random.permutation(len(self.data)):
                        yield self.data[i]

            data = generator()
        else:
            data = iter(self.data)

        d_current = next(data)
        for d_next in data:
            yield False, d_current
            d_current = d_next

        yield True, d_current

    def __iter__(self, random=False):
        raise NotImplementedError

    def for_fit(self, random=True):
        while True:
            yield from self.__iter__(random)

    def to_dataset(self, types, shapes, names=None, padded_batch=False):
        """转为tf.data.Dataset格式
        如果传入names的话，自动把数据包装成dict形式。
        """
        if names is None:

            generator = self.for_fit

        else:

            if isinstance(names, str):
                warps = lambda k, v: {k: v}
            elif isinstance(names[0], str):
                warps = lambda k, v: dict(zip(k, v))
            else:
                warps = lambda k, v: tuple(dict(zip(i, j)) for i, j in zip(k, v))

            def generator():
                for d in self.for_fit():
                    yield warps(names, d)

            types = warps(names, types)
            shapes = warps(names, shapes)

        if padded_batch:
            dataset = tf.data.Dataset.from_generator(generator, output_types=types)
            dataset = dataset.padded_batch(self.batch_size, shapes)
        else:
            dataset = tf.data.Dataset.from_generator(
                generator, output_types=types, output_shapes=shapes
            )
            dataset = dataset.batch(self.batch_size)

        return dataset


class TransformerTextClassificationDataGenerator(DataGenerator):
    def __init__(self, labels_num, *args, tokenizer=None, maxlen=128, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.labels_num = labels_num

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, item in self.sample(random):
            token_ids, segment_ids = self.tokenizer.encode(
                item["text"], maxlen=self.maxlen
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(item["label"])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = np.eye(self.labels_num)[np.array(batch_labels)]
                batch_labels = sequence_padding(batch_labels)
                yield {
                    "Input-Token": batch_token_ids,
                    "Input-Segment": batch_segment_ids,
                }, batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


class TransformerTextSequenceLabelingDataGenerator(DataGenerator):
    def __init__(self, categories, *args, tokenizer=None, maxlen=128, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.categories = categories

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, d in self.sample(random):
            tokens = self.tokenizer.tokenize(d[0], maxlen=self.maxlen)
            mapping = self.tokenizer.rematch(d[0], tokens)
            start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
            end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}
            token_ids = self.tokenizer.tokens_to_ids(tokens)
            segment_ids = [0] * len(token_ids)
            labels = np.zeros(len(token_ids))
            for start, end, label in d[1:]:
                if start in start_mapping and end in end_mapping:
                    start = start_mapping[start]
                    end = end_mapping[end]
                    labels[start] = self.categories.index(label) * 2 + 1
                    labels[start + 1 : end + 1] = self.categories.index(label) * 2 + 2
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []

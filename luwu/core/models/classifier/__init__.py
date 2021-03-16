# -*- coding: utf-8 -*-
# @Date         : 2020-12-30
# @Author       : AaronJny
# @LastEditTime : 2021-03-16
# @FilePath     : /LuWu/luwu/core/models/classifier/__init__.py
# @Desc         :
import os
import random

import tensorflow as tf
from loguru import logger
from luwu.core.preprocess.data.data_generator import ImageClassifierDataGnenrator
from luwu.core.preprocess.image.load import (
    read_classify_dataset_from_dir,
    write_tfrecords_to_target_path,
)
from luwu.utils import file_util
import numpy as np


class LuwuImageClassifier:
    def __init__(
        self,
        origin_dataset_path: str = "",
        tfrecord_dataset_path: str = "",
        model_save_path: str = "",
        validation_split: float = 0.2,
        batch_size: int = 32,
        epochs: int = 30,
        project_id: int = 0,
        image_size: int = 224,
        do_fine_tune=False,
        with_image_net=True,
        **kwargs,
    ):
        """
        Args:
            origin_dataset_path (str): 处理前的数据集路径
            tfrecord_dataset_path (str): 处理后的数据集路径
            model_save_path (str): 模型保存路径
            validation_split (float): 验证集切割比例
            batch_size (int): mini batch 大小
            epochs (int): 训练epoch数
            project_id (int): 训练项目编号
            with_image_net (bool): 是否使用imagenet的均值初始化数据
        """
        self._call_code = ""
        self.project_id = project_id
        self.do_fine_tune = do_fine_tune
        self.with_image_net = with_image_net
        origin_dataset_path = file_util.abspath(origin_dataset_path)
        tfrecord_dataset_path = file_util.abspath(tfrecord_dataset_path)
        model_save_path = file_util.abspath(model_save_path)
        self.image_size = image_size
        self.origin_dataset_path = origin_dataset_path
        # 当未给定处理后数据集的路径时，默认保存到原始数据集相同路径
        if tfrecord_dataset_path:
            self.tfrecord_dataset_path = tfrecord_dataset_path
        else:
            self.tfrecord_dataset_path = origin_dataset_path
        # 当未给定模型保存路径时，默认保存到处理后数据集相同路径
        if self.project_id:
            self.project_save_name = f"luwu-classification-project-{self.project_id}"
        else:
            self.project_save_name = f"luwu-classification-project"
        if model_save_path:
            self.project_save_path = os.path.join(
                model_save_path, self.project_save_name
            )
        else:
            self.project_save_path = os.path.join(
                self.tfrecord_dataset_path, self.project_save_name
            )
        self.model_save_path = os.path.join(self.project_save_path, "best_weights.h5")
        self.validation_split = validation_split
        self.batch_size = batch_size
        self.epochs = epochs
        file_util.mkdirs(self.project_save_path)
        file_util.mkdirs(self.tfrecord_dataset_path)

    def build_model(self) -> tf.keras.Model:
        """构建模型

        Raises:
            NotImplementedError: 待实现具体方法
        """
        raise NotImplementedError

    @property
    def kaggle_envs(self):
        """返回使用kaggle运行时的需要的参数"""
        return {}

    def preprocess_dataset(self):
        """对数据集进行预处理"""
        # 读取原始数据
        data, classes_num_dict = read_classify_dataset_from_dir(
            self.origin_dataset_path
        )
        # 类别->编号的映射
        self.classes_num_dict = classes_num_dict
        # 编号->类别的映射
        self.classes_num_dict_rev = {
            value: key for key, value in self.classes_num_dict.items()
        }
        # 先判断tfrecord是否存在
        self.target_train_dataset_path = os.path.join(
            self.tfrecord_dataset_path, "train_dataset"
        )
        self.target_dev_dataset_path = os.path.join(
            self.tfrecord_dataset_path, "dev_dataset"
        )
        # if (
        #     os.path.exists(self.target_train_dataset_path)
        #     and os.path.exists(self.target_dev_dataset_path)
        #     and os.path.isfile(self.target_train_dataset_path)
        #     and os.path.isfile(self.target_dev_dataset_path)
        # ):
        #     logger.info("TFRecord数据集已存在。跳过！")
        # else:
        # 切分数据
        total = len(data)
        dev_nums = int(total * self.validation_split)
        dev_data = random.sample(data, dev_nums)
        train_data = list(set(data) - set(dev_data))
        np.random.shuffle(train_data)
        # np.random.shuffle(dev_data)
        del data

        # 制作tfrecord数据集
        write_tfrecords_to_target_path(
            train_data, len(classes_num_dict), self.target_train_dataset_path
        )
        write_tfrecords_to_target_path(
            dev_data, len(classes_num_dict), self.target_dev_dataset_path
        )
        # 读取tfrecord数据集
        self.train_dataset = ImageClassifierDataGnenrator(
            self.target_train_dataset_path,
            batch_size=self.batch_size,
            image_size=self.image_size,
            with_image_net=self.with_image_net,
        )
        self.dev_dataset = ImageClassifierDataGnenrator(
            self.target_dev_dataset_path,
            batch_size=self.batch_size,
            image_size=self.image_size,
            with_image_net=self.with_image_net,
        )

    def train(self):
        # callbacks
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            self.model_save_path, monitor="val_accuracy", save_best_only=True
        )
        # 训练
        self.model.fit(
            self.train_dataset.for_fit(),
            epochs=self.epochs,
            steps_per_epoch=self.train_dataset.steps,
            validation_data=self.dev_dataset.for_fit(),
            validation_steps=self.dev_dataset.steps,
            callbacks=[
                checkpoint,
            ],
        )

    def run(self):
        # 预处理数据集
        logger.info("正在预处理数据集...")
        self.preprocess_dataset()
        # 构建模型
        logger.info("正在构建模型...")
        self.build_model()
        # 训练模型
        logger.info("开始训练...")
        self.train()
        # 导出代码
        logger.info("导出代码...")
        self.save_code()
        logger.info("Done.")

    def generator_train_code(self):
        """导出模型定义和训练代码"""
        raise NotImplementedError

    def get_call_code(self):
        """返回模型定义和模型调用的代码"""
        if self._call_code:
            return self._call_code
        else:
            raise NotImplementedError

    def save_code(self):
        """导出模型定义和模型调用的代码"""
        raise NotImplementedError


from luwu.core.models.classifier.preset import (
    LuwuLeNetImageClassifier,
    LuwuDenseNet121ImageClassifier,
    LuwuDenseNet169ImageClassifier,
    LuwuDenseNet201ImageClassifier,
    LuwuInceptionResNetV2ImageClassifier,
    LuwuInceptionV3ImageClassifier,
    LuwuMobileNetImageClassifier,
    LuwuMobileNetV2ImageClassifier,
    LuwuMobileNetV3LargeImageClassifier,
    LuwuMobileNetV3SmallImageClassifier,
    LuwuNASNetLargeImageClassifier,
    LuwuNASNetMobileImageClassifier,
    LuwuResNet50ImageClassifier,
    LuwuResNet50V2ImageClassifier,
    LuwuResNet101ImageClassifier,
    LuwuResNet101V2ImageClassifier,
    LuwuResNet152ImageClassifier,
    LuwuResNet152V2ImageClassifier,
    LuwuVGG16ImageClassifier,
    LuwuVGG19ImageClassifier,
    LuwuXceptionImageClassifier,
    LuwuEfficientNetB0ImageClassifier,
    LuwuEfficientNetB1ImageClassifier,
    LuwuEfficientNetB2ImageClassifier,
    LuwuEfficientNetB3ImageClassifier,
    LuwuEfficientNetB4ImageClassifier,
    LuwuEfficientNetB5ImageClassifier,
    LuwuEfficientNetB6ImageClassifier,
    LuwuEfficientNetB7ImageClassifier,
)

__all__ = [
    "LuwuImageClassifier",
    "LuwuPreTrainedImageClassifier",
    "LuwuLeNetImageClassifier",
    "LuwuDenseNet121ImageClassifier",
    "LuwuDenseNet169ImageClassifier",
    "LuwuDenseNet201ImageClassifier",
    "LuwuVGG16ImageClassifier",
    "LuwuVGG19ImageClassifier",
    "LuwuMobileNetImageClassifier",
    "LuwuMobileNetV2ImageClassifier",
    "LuwuInceptionResNetV2ImageClassifier",
    "LuwuInceptionV3ImageClassifier",
    "LuwuNASNetMobileImageClassifier",
    "LuwuNASNetLargeImageClassifier",
    "LuwuResNet50ImageClassifier",
    "LuwuResNet50V2ImageClassifier",
    "LuwuResNet101ImageClassifier",
    "LuwuResNet101V2ImageClassifier",
    "LuwuResNet152ImageClassifier",
    "LuwuResNet152V2ImageClassifier",
    "LuwuMobileNetV3SmallImageClassifier",
    "LuwuMobileNetV3LargeImageClassifier",
    "LuwuXceptionImageClassifier",
    "LuwuEfficientNetB0ImageClassifier",
    "LuwuEfficientNetB1ImageClassifier",
    "LuwuEfficientNetB2ImageClassifier",
    "LuwuEfficientNetB3ImageClassifier",
    "LuwuEfficientNetB4ImageClassifier",
    "LuwuEfficientNetB5ImageClassifier",
    "LuwuEfficientNetB6ImageClassifier",
    "LuwuEfficientNetB7ImageClassifier",
]

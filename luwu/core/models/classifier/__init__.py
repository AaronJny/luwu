# -*- coding: utf-8 -*-
# @Date         : 2020-12-30
# @Author       : AaronJny
# @LastEditTime : 2021-01-28
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


class LuwuImageClassifier:
    def __init__(
        self,
        origin_dataset_path: str = "",
        target_dataset_path: str = "",
        model_save_path: str = "",
        validation_split: float = 0.2,
        batch_size: int = 32,
        epochs: int = 30,
        project_id: int = 0,
        **kwargs,
    ):
        """
        Args:
            origin_dataset_path (str): 处理前的数据集路径
            target_dataset_path (str): 处理后的数据集路径
            model_save_path (str): 模型保存路径
            validation_split (float): 验证集切割比例
            batch_size (int): mini batch 大小
            epochs (int): 训练epoch数
            project_id (int): 训练项目编号
        """
        self._call_code = ""
        self.project_id = project_id
        self.origin_dataset_path = origin_dataset_path
        # 当未给定处理后数据集的路径时，默认保存到原始数据集相同路径
        if target_dataset_path:
            self.target_dataset_path = target_dataset_path
        else:
            self.target_dataset_path = origin_dataset_path
        # 当未给定模型保存路径时，默认保存到处理后数据集相同路径
        if self.project_id:
            model_file_name = f"best_weights_project_{self.project_id}.h5"
        else:
            model_file_name = "best_weights.h5"
        if model_save_path:
            self.model_save_path = os.path.join(model_save_path, model_file_name)
        else:
            self.model_save_path = os.path.join(
                self.target_dataset_path, model_file_name
            )
        self.validation_split = validation_split
        self.batch_size = batch_size
        self.epochs = epochs

    def build_model(self) -> tf.keras.Model:
        """构建模型

        Raises:
            NotImplementedError: 待实现具体方法
        """
        raise NotImplementedError

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
        # 切分数据
        total = len(data)
        dev_nums = int(total * self.validation_split)
        dev_data = random.sample(data, dev_nums)
        train_data = list(set(data) - set(dev_data))
        del data
        # 制作tfrecord数据集
        self.target_train_dataset_path = os.path.join(
            self.target_dataset_path, "train_dataset"
        )
        self.target_dev_dataset_path = os.path.join(
            self.target_dataset_path, "dev_dataset"
        )
        write_tfrecords_to_target_path(
            train_data, len(classes_num_dict), self.target_train_dataset_path
        )
        write_tfrecords_to_target_path(
            dev_data, len(classes_num_dict), self.target_dev_dataset_path
        )
        # 读取tfrecord数据集
        self.train_dataset = ImageClassifierDataGnenrator(
            self.target_train_dataset_path, batch_size=self.batch_size
        )
        self.dev_dataset = ImageClassifierDataGnenrator(
            self.target_dev_dataset_path, batch_size=self.batch_size, shuffle=False
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
)

__all__ = [
    "LuwuImageClassifier",
    "LuwuPreTrainedImageClassifier",
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
]

# -*- coding: utf-8 -*-
# @Date         : 2020-12-30
# @Author       : AaronJny
# @LastEditTime : 2021-04-15
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
        validation_dataset_path: str = "",
        test_dataset_path: str = "",
        tfrecord_dataset_path: str = "",
        model_save_path: str = "",
        validation_split: float = 0.1,
        test_split: float = 0.1,
        batch_size: int = 32,
        epochs: int = 30,
        learning_rate: float = 0.01,
        project_id: int = 0,
        image_size: int = 224,
        do_fine_tune=False,
        with_image_net=True,
        optimizer: str = "Adam",
        freeze_epochs_ratio: float = 0.1,
        image_augmentation_random_flip_horizontal: bool = False,
        image_augmentation_random_flip_vertival: bool = False,
        image_augmentation_random_crop: bool = False,
        image_augmentation_random_brightness: bool = False,
        image_augmentation_random_hue: bool = False,
        **kwargs,
    ):
        """
        Args:
            origin_dataset_path (str): 处理前的数据集路径
            validation_dataset_path (str): 验证数据集路径。如不指定，
                    则从origin_dataset_path中进行切分。
            test_dataset_path (str): 测试数据集路径。如不指定，则从
                                    origin_dataset_path中进行切分。
            tfrecord_dataset_path (str): 处理后的数据集路径
            model_save_path (str): 模型保存路径
            validation_split (float): 验证集切割比例
            test_split (float): 测试集切割比例
            batch_size (int): mini batch 大小
            learning_rate (float): 学习率大小
            epochs (int): 训练epoch数
            project_id (int): 训练项目编号
            with_image_net (bool): 是否使用imagenet的均值初始化数据
            optimizer (str): 优化器类别
            freeze_epochs_ratio (float): 当进行fine_tune时，会先冻结预训练模型进行训练一定epochs，
                                        再解冻全部参数训练一定epochs，此参数表示冻结训练epochs占
                                        全部epochs的比例（此参数仅当 do_fine_tune = True 时有效）。
                                        默认 0.1（当总epochs>1时，只要设置了比例，至少会训练一个epoch）
            image_augmentation_random_flip_horizontal (bool): 数据增强选项，是否做随机左右镜像。默认False.
            image_augmentation_random_flip_vertival (bool): 数据增强选项，是否做随机上下镜像。默认False.
            image_augmentation_random_crop (bool): 数据增强选项，是否做随机剪裁，剪裁尺寸为原来比例的0.9。默认False.
            image_augmentation_random_brightness (bool): 数据增强选项，是否做随机饱和度调节。默认False.
            image_augmentation_random_hue (bool): 数据增强选项，是否做随机色调调节。默认False.
        """
        self._call_code = ""
        self.project_id = project_id
        self.do_fine_tune = do_fine_tune
        self.with_image_net = with_image_net
        self.learning_rate = learning_rate
        self.freeze_epochs_ratio = freeze_epochs_ratio
        self.image_augmentation_random_flip_horizontal = (
            image_augmentation_random_flip_horizontal
        )
        self.image_augmentation_random_flip_vertival = (
            image_augmentation_random_flip_vertival
        )
        self.image_augmentation_random_crop = image_augmentation_random_crop
        self.image_augmentation_random_brightness = image_augmentation_random_brightness
        self.image_augmentation_random_hue = image_augmentation_random_hue
        self.optimizer_cls = self.get_optimizer_cls(optimizer)
        origin_dataset_path = file_util.abspath(origin_dataset_path)
        tfrecord_dataset_path = file_util.abspath(tfrecord_dataset_path)
        model_save_path = file_util.abspath(model_save_path)
        self.image_size = image_size
        self.origin_dataset_path = origin_dataset_path
        if validation_dataset_path:
            self.validation_dataset_path = file_util.abspath(validation_dataset_path)
        else:
            self.validation_dataset_path = validation_dataset_path
        if test_dataset_path:
            self.test_dataset_path = file_util.abspath(test_dataset_path)
        else:
            self.test_dataset_path = test_dataset_path
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
        self.test_split = test_split
        self.batch_size = batch_size
        self.epochs = epochs
        file_util.mkdirs(self.project_save_path)
        file_util.mkdirs(self.tfrecord_dataset_path)

    def get_optimizer_cls(self, optimizer_cls):
        optimizer_list = [
            "Adam",
            "Adamax",
            "Adagrad",
            "Nadam",
            "Adadelta",
            "SGD",
            "RMSprop",
        ]
        if isinstance(optimizer_cls, str):
            if optimizer_cls and optimizer_cls in optimizer_list:
                return getattr(tf.keras.optimizers, optimizer_cls)
        if issubclass(optimizer_cls, tf.keras.optimizers.Optimizer):
            return optimizer_cls
        raise Exception(f"指定的 Optimizer 类别不正确！{optimizer_cls}")

    def define_model(self) -> tf.keras.Model:
        """
        定义模型
        """
        raise NotImplementedError

    def define_optimizer(self):
        """
        定义优化器
        """
        self.model.compile(
            optimizer=self.optimizer_cls(learning_rate=self.learning_rate),
            loss=tf.keras.losses.categorical_crossentropy,
            metrics=["accuracy"],
        )

    def build_model(self) -> tf.keras.Model:
        """构建模型

        Raises:
            NotImplementedError: 待实现具体方法
        """
        self.model = self.define_model()
        self.define_optimizer()
        return self.model

    @property
    def kaggle_envs(self):
        """返回使用kaggle运行时的需要的参数"""
        return {}

    def load_data_from_dir(self, dirpath, classes_num_dict=None):
        data, classes_num_dict = read_classify_dataset_from_dir(
            dirpath, classes_num_dict=classes_num_dict
        )
        return data, classes_num_dict

    def preprocess_dataset(self):
        """对数据集进行预处理"""
        # 读取原始数据
        data, classes_num_dict = read_classify_dataset_from_dir(
            self.origin_dataset_path
        )
        # 切分数据集
        total = len(data)
        # 判断有没有指定验证集
        if self.validation_dataset_path:
            # 如果指定了，就从指定路径读取
            dev_data, classes_num_dict = read_classify_dataset_from_dir(
                self.validation_dataset_path, classes_num_dict=classes_num_dict
            )
        else:
            dev_nums = int(total * self.validation_split)
            dev_data = random.sample(data, dev_nums)
        # 判断有没有指定测试集
        if self.test_dataset_path:
            # 如果制定了，就从指定路径读取
            test_data, classes_num_dict = read_classify_dataset_from_dir(
                self.test_dataset_path, classes_num_dict=classes_num_dict
            )
        else:
            test_nums = int(total * self.test_split)
            test_data = random.sample(data, test_nums)
        train_data = list(set(data) - set(dev_data) - set(test_data))
        np.random.shuffle(train_data)
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
        if test_data:
            self.target_test_dataset_path = os.path.join(
                self.tfrecord_dataset_path, "dev_dataset"
            )
        else:
            self.target_test_dataset_path = None
        # if (
        #     os.path.exists(self.target_train_dataset_path)
        #     and os.path.exists(self.target_dev_dataset_path)
        #     and os.path.isfile(self.target_train_dataset_path)
        #     and os.path.isfile(self.target_dev_dataset_path)
        # ):
        #     logger.info("TFRecord数据集已存在。跳过！")
        # else:

        del data

        # 制作tfrecord数据集
        write_tfrecords_to_target_path(
            train_data, len(classes_num_dict), self.target_train_dataset_path
        )
        write_tfrecords_to_target_path(
            dev_data, len(classes_num_dict), self.target_dev_dataset_path
        )
        if self.target_test_dataset_path:
            write_tfrecords_to_target_path(
                test_data, len(classes_num_dict), self.target_test_dataset_path
            )
        # 读取tfrecord数据集
        self.train_dataset = ImageClassifierDataGnenrator(
            self.target_train_dataset_path,
            batch_size=self.batch_size,
            image_size=self.image_size,
            with_image_net=self.with_image_net,
            image_augmentation_random_flip_horizontal=self.image_augmentation_random_flip_horizontal,
            image_augmentation_random_flip_vertival=self.image_augmentation_random_flip_vertival,
            image_augmentation_random_crop=self.image_augmentation_random_crop,
            image_augmentation_random_brightness=self.image_augmentation_random_brightness,
            image_augmentation_random_hue=self.image_augmentation_random_hue,
        )
        self.dev_dataset = ImageClassifierDataGnenrator(
            self.target_dev_dataset_path,
            batch_size=self.batch_size,
            image_size=self.image_size,
            with_image_net=self.with_image_net,
            image_augmentation_random_flip_horizontal=self.image_augmentation_random_flip_horizontal,
            image_augmentation_random_flip_vertival=self.image_augmentation_random_flip_vertival,
            image_augmentation_random_crop=self.image_augmentation_random_crop,
            image_augmentation_random_brightness=self.image_augmentation_random_brightness,
            image_augmentation_random_hue=self.image_augmentation_random_hue,
        )
        if self.target_test_dataset_path:
            self.test_dataset = ImageClassifierDataGnenrator(
                self.target_test_dataset_path,
                batch_size=self.batch_size,
                image_size=self.image_size,
                with_image_net=self.with_image_net,
                image_augmentation_random_flip_horizontal=self.image_augmentation_random_flip_horizontal,
                image_augmentation_random_flip_vertival=self.image_augmentation_random_flip_vertival,
                image_augmentation_random_crop=self.image_augmentation_random_crop,
                image_augmentation_random_brightness=self.image_augmentation_random_brightness,
                image_augmentation_random_hue=self.image_augmentation_random_hue,
            )
        else:
            self.test_dataset = None

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
        logger.info("在测试集上进行验证...")
        if self.test_dataset:
            evaluate_dataset = self.test_dataset
        else:
            evaluate_dataset = self.dev_dataset
        logger.info(
            self.model.evaluate(
                evaluate_dataset.for_fit(), steps=evaluate_dataset.steps
            )
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
    LuwuPreTrainedImageClassifier,
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

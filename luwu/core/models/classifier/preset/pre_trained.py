# -*- coding: utf-8 -*-
# @Date         : 2021-01-21
# @Author       : AaronJny
# @LastEditTime : 2021-04-06
# @FilePath     : /LuWu/luwu/core/models/classifier/preset/pre_trained.py
# @Desc         : 封装tf.keras里设置的预训练模型，并对外提供支持
import os
from abc import ABC

import tensorflow as tf
from jinja2 import Template
from luwu.core.models.classifier import LuwuImageClassifier
from loguru import logger


class LuwuPreTrainedImageClassifier(LuwuImageClassifier, ABC):
    def __init__(self, net_name, *args, **kwargs):
        super(LuwuPreTrainedImageClassifier, self).__init__(*args, **kwargs)
        self.net_name = net_name

    def define_model(self) -> tf.keras.Model:
        pre_trained_net: tf.keras.Model = getattr(
            tf.keras.applications, self.net_name
        )()
        pre_trained_net.trainable = False
        # 记录densenet
        self.pre_trained_net = pre_trained_net
        model = tf.keras.Sequential(
            [
                pre_trained_net,
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(len(self.classes_num_dict), activation="softmax"),
            ]
        )
        return model

    def train(self):
        # callbacks
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            self.model_save_path, monitor="val_accuracy", save_best_only=True
        )
        if self.do_fine_tune and self.freeze_epochs_ratio:
            # 如果选择了fine tune，则至少冻结训练一个epoch
            pre_train_epochs = max(1, int(self.freeze_epochs_ratio * self.epochs))
        else:
            pre_train_epochs = 0
        train_epochs = self.epochs - pre_train_epochs
        if pre_train_epochs:
            logger.info(
                f"分两步训练，冻结训练{pre_train_epochs}个epochs，解冻训练{train_epochs}个epochs..."
            )
        # 训练
        if pre_train_epochs:
            logger.info("冻结 pre-trained 模型，开始预训练 ...")
            self.model.fit(
                self.train_dataset.for_fit(),
                initial_epoch=0,
                epochs=pre_train_epochs,
                steps_per_epoch=self.train_dataset.steps,
                validation_data=self.dev_dataset.for_fit(),
                validation_steps=self.dev_dataset.steps,
                callbacks=[
                    checkpoint,
                ],
            )
        if train_epochs:
            logger.info("解冻 pre-trained 模型，继续训练 ...")
            self.pre_trained_net.trainable = True
            self.model.fit(
                self.train_dataset.for_fit(),
                initial_epoch=pre_train_epochs,
                epochs=self.epochs,
                steps_per_epoch=self.train_dataset.steps,
                validation_data=self.dev_dataset.for_fit(),
                validation_steps=self.dev_dataset.steps,
                callbacks=[
                    checkpoint,
                ],
            )
        logger.info("在测试集上进行验证...")
        # self.model.load_weights(self.model_save_path, by_name=True)
        logger.info(
            self.model.evaluate(
                self.dev_dataset.for_fit(), steps=self.dev_dataset.steps
            )
        )

    def get_call_code(self):
        """返回模型定义和模型调用的代码"""
        if not self._call_code:
            template_path = os.path.join(
                os.path.dirname(__file__), "templates/LuwuPreTrainedImageClassifier.txt"
            )
            with open(template_path, "r") as f:
                text = f.read()
            data = {
                "net_name": self.net_name,
                "num_classes": len(self.classes_num_dict),
                "num_classes_map": str(self.classes_num_dict_rev),
                "image_size": self.image_size,
                "model_path": self.model_save_path,
                "data_preprocess_template": self.train_dataset.generate_preprocess_code(),
            }
            template = Template(text)
            code = template.render(**data)
            self._call_code = code
        return self._call_code

    def save_code(self):
        """导出模型定义和模型调用的代码"""
        code = self.get_call_code()
        code_file_name = "luwu-code.py"
        code_path = os.path.join(self.project_save_path, code_file_name)
        with open(code_path, "w") as f:
            f.write(code)


class LuwuLeNetImageClassifier(LuwuPreTrainedImageClassifier):
    def __init__(self, *args, **kwargs):
        kwargs["net_name"] = "LeNet"
        kwargs["with_image_net"] = False
        kwargs["do_fine_tune"] = False
        kwargs["image_size"] = 32
        super(LuwuLeNetImageClassifier, self).__init__(*args, **kwargs)

    def define_model(self) -> tf.keras.Model:
        # todo: 为模型增加dropout和正则，以适当减轻过拟合
        model = tf.keras.Sequential(
            [
                tf.keras.Input(shape=(self.image_size, self.image_size, 3)),
                tf.keras.layers.Conv2D(6, (5, 5), padding="same"),
                # 添加BN层，将数据调整为均值0，方差1
                tf.keras.layers.BatchNormalization(),
                # 最大池化层，池化后图片长宽减半
                tf.keras.layers.MaxPooling2D((2, 2), 2),
                # relu激活层
                tf.keras.layers.ReLU(),
                # 第二个卷积层
                tf.keras.layers.Conv2D(16, (5, 5)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D((2, 2), 2),
                tf.keras.layers.ReLU(),
                # 将节点展平为(None,-1)的形式，以作为全连接层的输入
                tf.keras.layers.Flatten(),
                # 第一个全连接层，120个节点
                tf.keras.layers.Dense(120),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                # 第二个全连接层
                tf.keras.layers.Dense(84),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                # tf.keras.layers.Dropout(0.3),
                # 输出层，使用softmax激活
                tf.keras.layers.Dense(len(self.classes_num_dict), activation="softmax"),
            ]
        )
        # use_regularizer = True
        # if use_regularizer:
        #     for layer in model.layers:
        #         if hasattr(layer, "kernel_regularizer"):
        #             layer.kernel_regularizer = tf.keras.regularizers.l2(0.01)
        return model

    def train(self):
        # callbacks
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            self.model_save_path, monitor="val_accuracy", save_best_only=True
        )
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
        logger.info("加载最优参数，输出验证集结果 ...")
        self.model.load_weights(self.model_save_path)
        self.model.evaluate(self.dev_dataset.for_fit(), steps=self.dev_dataset.steps)


class LuwuDenseNet121ImageClassifier(LuwuPreTrainedImageClassifier):
    def __init__(self, net_name="DenseNet121", **kwargs):
        super(LuwuDenseNet121ImageClassifier, self).__init__(net_name, **kwargs)


class LuwuDenseNet169ImageClassifier(LuwuPreTrainedImageClassifier):
    def __init__(self, net_name="DenseNet169", **kwargs):
        super(LuwuDenseNet169ImageClassifier, self).__init__(net_name, **kwargs)


class LuwuDenseNet201ImageClassifier(LuwuPreTrainedImageClassifier):
    def __init__(self, net_name="DenseNet201", **kwargs):
        super(LuwuDenseNet201ImageClassifier, self).__init__(net_name, **kwargs)


class LuwuVGG16ImageClassifier(LuwuPreTrainedImageClassifier):
    def __init__(self, net_name="VGG16", **kwargs):
        super(LuwuVGG16ImageClassifier, self).__init__(net_name, **kwargs)


class LuwuVGG19ImageClassifier(LuwuPreTrainedImageClassifier):
    def __init__(self, net_name="VGG19", **kwargs):
        super(LuwuVGG19ImageClassifier, self).__init__(net_name, **kwargs)


class LuwuMobileNetImageClassifier(LuwuPreTrainedImageClassifier):
    def __init__(self, net_name="MobileNet", **kwargs):
        super(LuwuMobileNetImageClassifier, self).__init__(net_name, **kwargs)


class LuwuMobileNetV2ImageClassifier(LuwuPreTrainedImageClassifier):
    def __init__(self, net_name="MobileNetV2", **kwargs):
        super(LuwuMobileNetV2ImageClassifier, self).__init__(net_name, **kwargs)


class LuwuInceptionResNetV2ImageClassifier(LuwuPreTrainedImageClassifier):
    def __init__(self, net_name="InceptionResNetV2", **kwargs):
        super(LuwuInceptionResNetV2ImageClassifier, self).__init__(net_name, **kwargs)


class LuwuInceptionV3ImageClassifier(LuwuPreTrainedImageClassifier):
    def __init__(self, net_name="InceptionV3", **kwargs):
        super(LuwuInceptionV3ImageClassifier, self).__init__(net_name, **kwargs)


class LuwuNASNetMobileImageClassifier(LuwuPreTrainedImageClassifier):
    def __init__(self, net_name="NASNetMobile", **kwargs):
        super(LuwuNASNetMobileImageClassifier, self).__init__(net_name, **kwargs)


class LuwuNASNetLargeImageClassifier(LuwuPreTrainedImageClassifier):
    def __init__(self, net_name="NASNetLarge", **kwargs):
        super(LuwuNASNetLargeImageClassifier, self).__init__(net_name, **kwargs)


class LuwuResNet50ImageClassifier(LuwuPreTrainedImageClassifier):
    def __init__(self, net_name="ResNet50", **kwargs):
        super(LuwuResNet50ImageClassifier, self).__init__(net_name, **kwargs)


class LuwuResNet50V2ImageClassifier(LuwuPreTrainedImageClassifier):
    def __init__(self, net_name="ResNet50V2", **kwargs):
        super(LuwuResNet50V2ImageClassifier, self).__init__(net_name, **kwargs)


class LuwuResNet101ImageClassifier(LuwuPreTrainedImageClassifier):
    def __init__(self, net_name="ResNet101", **kwargs):
        super(LuwuResNet101ImageClassifier, self).__init__(net_name, **kwargs)


class LuwuResNet101V2ImageClassifier(LuwuPreTrainedImageClassifier):
    def __init__(self, net_name="ResNet101V2", **kwargs):
        super(LuwuResNet101V2ImageClassifier, self).__init__(net_name, **kwargs)


class LuwuResNet152ImageClassifier(LuwuPreTrainedImageClassifier):
    def __init__(self, net_name="ResNet152", **kwargs):
        super(LuwuResNet152ImageClassifier, self).__init__(net_name, **kwargs)


class LuwuResNet152V2ImageClassifier(LuwuPreTrainedImageClassifier):
    def __init__(self, net_name="ResNet152V2", **kwargs):
        super(LuwuResNet152V2ImageClassifier, self).__init__(net_name, **kwargs)


class LuwuMobileNetV3SmallImageClassifier(LuwuPreTrainedImageClassifier):
    def __init__(self, net_name="MobileNetV3Small", **kwargs):
        super(LuwuMobileNetV3SmallImageClassifier, self).__init__(net_name, **kwargs)


class LuwuMobileNetV3LargeImageClassifier(LuwuPreTrainedImageClassifier):
    def __init__(self, net_name="MobileNetV3Large", **kwargs):
        super(LuwuMobileNetV3LargeImageClassifier, self).__init__(net_name, **kwargs)


class LuwuXceptionImageClassifier(LuwuPreTrainedImageClassifier):
    def __init__(self, net_name="Xception", **kwargs):
        super(LuwuXceptionImageClassifier, self).__init__(net_name, **kwargs)


class LuwuEfficientNetB0ImageClassifier(LuwuPreTrainedImageClassifier):
    def __init__(self, net_name="EfficientNetB0", **kwargs):
        super(LuwuEfficientNetB0ImageClassifier, self).__init__(net_name, **kwargs)


class LuwuEfficientNetB1ImageClassifier(LuwuPreTrainedImageClassifier):
    def __init__(self, net_name="EfficientNetB1", **kwargs):
        super(LuwuEfficientNetB1ImageClassifier, self).__init__(net_name, **kwargs)


class LuwuEfficientNetB2ImageClassifier(LuwuPreTrainedImageClassifier):
    def __init__(self, net_name="EfficientNetB2", **kwargs):
        super(LuwuEfficientNetB2ImageClassifier, self).__init__(net_name, **kwargs)


class LuwuEfficientNetB3ImageClassifier(LuwuPreTrainedImageClassifier):
    def __init__(self, net_name="EfficientNetB3", **kwargs):
        super(LuwuEfficientNetB3ImageClassifier, self).__init__(net_name, **kwargs)


class LuwuEfficientNetB4ImageClassifier(LuwuPreTrainedImageClassifier):
    def __init__(self, net_name="EfficientNetB4", **kwargs):
        super(LuwuEfficientNetB4ImageClassifier, self).__init__(net_name, **kwargs)


class LuwuEfficientNetB5ImageClassifier(LuwuPreTrainedImageClassifier):
    def __init__(self, net_name="EfficientNetB5", **kwargs):
        super(LuwuEfficientNetB5ImageClassifier, self).__init__(net_name, **kwargs)


class LuwuEfficientNetB6ImageClassifier(LuwuPreTrainedImageClassifier):
    def __init__(self, net_name="EfficientNetB6", **kwargs):
        super(LuwuEfficientNetB6ImageClassifier, self).__init__(net_name, **kwargs)


class LuwuEfficientNetB7ImageClassifier(LuwuPreTrainedImageClassifier):
    def __init__(self, net_name="EfficientNetB7", **kwargs):
        super(LuwuEfficientNetB7ImageClassifier, self).__init__(net_name, **kwargs)
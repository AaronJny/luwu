# -*- coding: utf-8 -*-
# @Date         : 2021-01-21
# @Author       : AaronJny
# @LastEditTime : 2021-01-22
# @FilePath     : /LuWu/luwu/core/models/classifier/preset/pre_trained.py
# @Desc         : 封装tf.keras里设置的预训练模型，并对外提供支持
import os

import tensorflow as tf
from jinja2 import Template
from luwu.core.models.classifier import LuwuImageClassifier


class LuwuPreTrainedImageClassifier(LuwuImageClassifier):
    def __init__(self, net_name, *args, **kwargs):
        super(LuwuPreTrainedImageClassifier, self).__init__(*args, **kwargs)
        self.net_name = net_name

    def build_model(self):
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
                tf.keras.layers.Dense(120, activation="relu"),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(len(self.classes_num_dict), activation="softmax"),
            ]
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.categorical_crossentropy,
            metrics=["accuracy"],
        )
        self.model = model
        return model

    def generate_code(self):
        """导出模型定义和模型调用的代码"""
        template_path = os.path.join(
            os.path.dirname(__file__), "templates/LuwuPreTrainedImageClassifier.txt"
        )
        with open(template_path, "r") as f:
            text = f.read()
        data = {
            "net_name": self.net_name,
            "num_classes": len(self.classes_num_dict),
            "num_classes_map": str(self.classes_num_dict_rev),
            "model_path": self.model_save_path,
            "data_preprocess_template": self.train_dataset.generate_preprocess_code(),
        }
        template = Template(text)
        code = template.render(**data)
        code_path = os.path.join(os.path.dirname(self.model_save_path), "luwu-code.py")
        with open(code_path, "w") as f:
            f.write(code)


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
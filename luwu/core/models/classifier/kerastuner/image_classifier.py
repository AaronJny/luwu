# -*- coding: utf-8 -*-
# @Date         : 2021-01-07
# @Author       : AaronJny
# @LastEditTime : 2021-01-28
# @FilePath     : /app/luwu/core/models/classifier/kerastuner/image_classifier.py
# @Desc         :
"""
这是还在规划中的功能，里面的代码目前都是无用的。
TODO:增加KerasTuner相关功能
"""
import tensorflow.keras as keras
from luwu.core.preprocess.image.load import (
    read_classify_dataset_from_dir,
    write_tfrecords_to_target_path,
)
from luwu.core.preprocess.image.process import (
    extract_image_and_label_from_record,
    normalized_image,
)
from tensorflow.keras import layers
import tensorflow as tf
from kerastuner.tuners.randomsearch import RandomSearch
import random
import os
from luwu.core.preprocess.data.data_generator import ImageClassifierDataGnenrator


class LuWuKerasTunerImageClassifier:
    def __init__(
        self,
        origin_dataset_path: str,
        target_dataset_path: str,
        max_trials: int = 20,
        executions_per_trial: int = 3,
        epochs: int = 10,
        batch_size: int = 4,
        validation_split: float = 0.2,
    ):
        self.origin_dataset_path = origin_dataset_path
        self.target_dataset_path = target_dataset_path
        self.batch_size = batch_size
        self.max_trials = max_trials
        self.executions_per_trial = executions_per_trial
        self.epochs = epochs
        self.validation_split = validation_split

    def _preprocess_dataset(self, dataset_path):
        dataset = tf.data.TFRecordDataset(dataset_path)
        dataset = dataset.map(extract_image_and_label_from_record)
        dataset = dataset.map(normalized_image)
        dataset = dataset.batch(self.batch_size)
        # dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset

    def preprocess_dataset(self):
        """对数据进行预处理"""
        # 读取原始数据
        data, classes_num_dict = read_classify_dataset_from_dir(
            self.origin_dataset_path
        )
        self.classes_num_dict = classes_num_dict
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

    def build_model(self, hp):
        model = keras.Sequential()
        model.add(layers.Input((224, 224, 3)))
        dropout_rate = hp.Float("dropout_rate", 0.1, 0.4, 0.05)
        conv_num_layers = hp.Int("conv_num_layers", 2, 3)
        dense_num_layers = hp.Int("dense_num_layers", 0, 1)
        for i in range(conv_num_layers):
            model.add(
                layers.Conv2D(
                    filters=hp.Int(f"filters_{i}", 8, 64, 8),
                    kernel_size=hp.Int(f"kernel_{i}", 3, 3, 2),
                    padding=hp.Choice(f"padding_{i}", ["same", "valid"]),
                    activation="relu",
                )
            )
            model.add(layers.Dropout(dropout_rate))
            if hp.Choice(f"maxpooling_{i}", [True, False]):
                model.add(layers.MaxPooling2D((2, 2), 2))
        model.add(layers.Flatten())
        for i in range(dense_num_layers):
            model.add(
                layers.Dense(units=hp.Int(f"units_{i}", 20, 60), activation="relu")
            )
        model.add(layers.Dense(len(self.classes_num_dict), activation="softmax"))
        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])
            ),
            loss=keras.losses.categorical_crossentropy,
            metrics=["accuracy"],
        )
        return model

    def search(self):
        tuner = RandomSearch(
            self.build_model,
            objective="val_accuracy",
            max_trials=self.max_trials,
            executions_per_trial=self.executions_per_trial,
            directory="test_dir",
        )

        tuner.search_space_summary()

        tuner.search(
            self.train_dataset.for_fit(),
            epochs=10,
            steps_per_epoch=self.train_dataset.steps,
            validation_data=self.dev_dataset.for_fit(),
            validation_steps=self.dev_dataset.steps,
        )
        tuner.results_summary()
        print("--" * 20)
        print(tuner.get_best_hyperparameters()[0].get_config())

    def valid(self):
        model = keras.Sequential(
            [
                layers.Input((224, 224, 3)),
                layers.Conv2D(6, 3, activation="relu"),
                layers.Conv2D(16, 3, activation="relu"),
                layers.Flatten(),
                layers.Dense(3, activation="softmax"),
            ]
        )
        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=tf.keras.losses.categorical_crossentropy,
            metrics=["accuracy"],
        )
        model.summary()
        return model

    def run(self):
        # 预处理数据集
        self.preprocess_dataset()
        self.search()


if __name__ == "__main__":
    LuWuKerasTunerImageClassifier(
        "/Users/aaron/code/pets_classifer/images",
        "/Users/aaron/code/pets_classifer/images",
    ).run()

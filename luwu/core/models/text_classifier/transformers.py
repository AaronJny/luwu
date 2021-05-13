# -*- coding: utf-8 -*-
# @Author       : AaronJny
# @LastEditTime : 2021-05-13
# @FilePath     : /LuWu/luwu/core/models/text_classifier/transformers.py
# @Desc         :
import os

os.environ["TF_KERAS"] = "1"
import json
import random
from collections import Counter
from glob import glob

import numpy as np
import tensorflow as tf
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.optimizers import extend_with_piecewise_linear_lr
from loguru import logger
from luwu.core.preprocess.data.data_generator import (
    TransformerTextClassificationDataGenerator,
)
from luwu.utils import file_util
from jinja2 import Template


class TransformerTextClassification(object):

    model_lang_weights_dict = {
        "bert_base": {
            "chinese": {
                "url": "https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip",
                "config_path": "bert_config.json",
                "checkpoint_path": "bert_model.ckpt",
                "dict_path": "vocab.txt",
            },
            "multilingual_cased": {
                "url": "https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip",
                "config_path": "bert_config.json",
                "checkpoint_path": "bert_model.ckpt",
                "dict_path": "vocab.txt",
            },
            "english_cased": {
                "url": "https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip",
                "config_path": "bert_config.json",
                "checkpoint_path": "bert_model.ckpt",
                "dict_path": "vocab.txt",
            },
            "english_uncased": {
                "url": "https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip",
                "config_path": "bert_config.json",
                "checkpoint_path": "bert_model.ckpt",
                "dict_path": "vocab.txt",
            },
        }
    }

    bert4keras_model_name_dict = {"bert": "bert", "bert_base": "bert"}

    def __init__(
        self,
        origin_dataset_path: str,
        validation_dataset_path: str = "",
        test_dataset_path: str = "",
        model_save_path: str = "",
        validation_split: float = 0.1,
        test_split: float = 0.1,
        batch_size: int = 32,
        epochs: int = 30,
        learning_rate: float = 0.01,
        project_id: int = 0,
        maxlen: int = 128,
        frezee_pre_trained_model=False,
        optimizer: str = "Adam",
        optimize_with_piecewise_linear_lr: bool = False,
        do_sample_balance: str = "",
        simplified_tokenizer: bool = False,
        pre_trained_model_type: str = "bert_base",
        language: str = "chinese",
        **kwargs,
    ):
        """
        Args:
            origin_dataset_path (str): 处理前的数据集路径
            validation_dataset_path (str): 验证数据集路径。如不指定，
                    则从origin_dataset_path中进行切分。
            test_dataset_path (str): 测试数据集路径。如不指定，则从
                                    origin_dataset_path中进行切分。
            model_save_path (str): 模型保存路径
            validation_split (float): 验证集切割比例
            test_split (float): 测试集切割比例
            batch_size (int): mini batch 大小
            learning_rate (float): 学习率大小
            epochs (int): 训练epoch数
            project_id (int): 训练项目编号
            maxlen (int, optional): 单个文本的最大长度. Defaults to 128.
            frezee_pre_trained_model (bool, optional): 在训练下游网络时，是否冻结预训练模型权重. Defaults to False.
            optimizer (str, optional): 优化器类别. Defaults to "Adam".
            optimize_with_piecewise_linear_lr (bool): 是否使用分段的线性学习率进行优化. 默认 False
            do_sample_balance (str): 是否对数据集做样本均衡，允许传递三个值，""表示不进行样本均衡，
                                "over"表示上采样（过采样），"under"表示下采样（欠采样）
            simplified_tokenizer (bool): 是否对分词器的词表进行精简，默认False
            pre_trained_model_type (str): 使用何种预训练模型
            language (str): 预训练语料的语言
        """
        self._call_code = ""
        self.project_id = project_id
        self.frezee_pre_trained_model = frezee_pre_trained_model
        self.learning_rate = learning_rate

        self.optimize_with_piecewise_linear_lr = optimize_with_piecewise_linear_lr
        self.optimizer_cls = self.get_optimizer_cls(optimizer)

        origin_dataset_path = file_util.abspath(origin_dataset_path)
        model_save_path = file_util.abspath(model_save_path)

        if do_sample_balance not in ("", "over", "under"):
            raise Exception("参数 do_sample_balance 必须为 '','over','under' 之一！")
        self.do_sample_balance = do_sample_balance

        self.simplified_tokenizer = simplified_tokenizer
        self.pre_trained_model_type = pre_trained_model_type
        self.language = language
        if self.pre_trained_model_type not in self.model_lang_weights_dict:
            raise Exception(
                f"指定模型 {self.pre_trained_model_type} 不存在！当前支持的模型为：{list(self.model_lang_weights_dict.keys())}"
            )
        if (
            self.language
            not in self.model_lang_weights_dict[self.pre_trained_model_type]
        ):
            languages = list(
                self.model_lang_weights_dict[self.pre_trained_model_type].keys()
            )
            raise Exception(
                f"指定语料 {self.language} 的预训练模型 {self.pre_trained_model_type} 不存在！支持的语料为：{languages}"
            )

        self.maxlen = maxlen
        self.origin_dataset_path = origin_dataset_path

        if validation_dataset_path:
            self.validation_dataset_path = file_util.abspath(validation_dataset_path)
        else:
            self.validation_dataset_path = validation_dataset_path

        if test_dataset_path:
            self.test_dataset_path = file_util.abspath(test_dataset_path)
        else:
            self.test_dataset_path = test_dataset_path

        # 当未给定模型保存路径时，默认保存到origin数据集相同路径
        if self.project_id:
            self.project_save_name = (
                f"luwu-text-classification-project-{self.project_id}"
            )
        else:
            self.project_save_name = f"luwu-text-classification-project"
        if model_save_path:
            self.project_save_path = os.path.join(
                model_save_path, self.project_save_name
            )
        else:
            self.project_save_path = os.path.join(
                os.path.dirname(self.origin_dataset_path), self.project_save_name
            )
        self.model_save_path = os.path.join(self.project_save_path, "best_weights.h5")

        self.validation_split = validation_split
        self.test_split = test_split
        self.batch_size = batch_size
        self.epochs = epochs

        file_util.mkdirs(self.project_save_path)

        self.model = None

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
                optimizer_cls = getattr(tf.keras.optimizers, optimizer_cls)
        if not issubclass(optimizer_cls, tf.keras.optimizers.Optimizer):
            raise Exception(f"指定的 Optimizer 类别不正确！{optimizer_cls}")
        if self.optimize_with_piecewise_linear_lr:
            optimizer_cls = extend_with_piecewise_linear_lr(optimizer_cls)
        return optimizer_cls

    def define_model(self) -> tf.keras.Model:
        """
        定义模型
        """
        model_name = self.bert4keras_model_name_dict[self.pre_trained_model_type]
        params = {
            "config_path": self.pre_trained_model_config_path,
            "checkpoint_path": self.pre_trained_model_checkpoint_path,
            "model": model_name,
            "return_keras_model": False,
        }
        if self.simplified_tokenizer:
            params["keep_tokens"] = self.keep_tokens
        self.transformer = build_transformer_model(**params)
        output = tf.keras.layers.Lambda(lambda x: x[:, 0], name="CLS-token")(
            self.transformer.model.output
        )
        output = tf.keras.layers.Dense(
            units=len(self.label_id_dict),
            activation="softmax",
            kernel_initializer=self.transformer.initializer,
        )(output)
        model = tf.keras.models.Model(self.transformer.model.input, output)
        model.summary()
        return model

    def define_optimizer(self):
        """
        定义优化器
        """
        params = {"learning_rate": self.learning_rate}
        if self.optimize_with_piecewise_linear_lr:
            params["lr_schedule"] = {1000: 1, 2000: 0.1}
        self.model.compile(
            optimizer=self.optimizer_cls(**params),
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

    def check_record(self, record):
        if not isinstance(record["text"], str):
            logger.warning(f"record['text'] 必须是 str 类型！当前类型：{type(record['text'])}")
            return False
        if not isinstance(record["labels"], list):
            logger.warning(
                f"record['labels'] 必须是 list 类型！当前类型：{type(record['labels'])}"
            )
            return False
        if len(record["labels"]) != 1:
            logger.warning(
                f"单文本分类任务中，record['labels'] 必须有且只有一个标签！当前标签数量：{len(record['labels'])}"
            )
            return False
        if not isinstance(record["labels"][0], str):
            logger.warning(
                f"record['labels'][0] 必须是 str 类型！当前类型：{type(record['labels'][0])}"
            )
            return False
        return True

    def load_data(self, filepath, label_id_dict=None):
        data = []
        counter = Counter()

        # 先遍历一边，确定全部标签，并将其编码固定下来（多次跑相同数据，编码的映射关系应当一致）
        for sub_filepath in glob(filepath):
            with open(sub_filepath, "r") as f:
                for line in f:
                    record = json.loads(line)
                    if self.check_record(record):
                        counter.update(record["labels"])

        labels = sorted(list(counter.keys()))
        if label_id_dict is None:
            label_id_dict = dict(zip(labels, range(len(labels))))
        counter = {label_id_dict[label]: count for label, count in counter.items()}

        # 再遍历第二版，处理数据
        for sub_filepath in glob(filepath):
            with open(sub_filepath, "r") as f:
                for line in f:
                    record = json.loads(line)
                    if self.check_record(record):
                        label = record["labels"][0]
                        label_id = label_id_dict[label]
                        data.append({"text": record["text"], "label": label_id})

        return data, counter, label_id_dict

    def guess(self, prob):
        """轮盘赌法确定概率检测是否通过"""
        prob = prob * 100 if prob < 1 else prob
        x = random.randint(0, 100)
        return x < prob

    def load_and_split_dataset(self):
        # 读取原始数据
        data, label_counter, label_id_dict = self.load_data(self.origin_dataset_path)
        # 切分数据集
        total = len(data)
        exclude_indexs = set()
        # 判断有没有指定验证集
        if self.validation_dataset_path:
            # 如果指定了，就从指定路径读取
            dev_data, _ = self.load_data(
                self.validation_dataset_path, label_id_dict=label_id_dict
            )
        else:
            dev_nums = int(total * self.validation_split)
            dev_data = []
            for _ in range(dev_nums):
                index = random.randint(0, total - 1)
                dev_data.append(data[index].copy())
        # 判断有没有指定测试集
        if self.test_dataset_path:
            # 如果制定了，就从指定路径读取
            test_data, _ = self.load_data(
                self.test_dataset_path, label_id_dict=label_id_dict
            )
        else:
            test_nums = int(total * self.test_split)
            test_data = []
            for _ in range(test_nums):
                index = random.randint(0, total - 1)
                test_data.append(data[index].copy())
        train_data = [
            data[index].copy() for index in range(total) if index not in exclude_indexs
        ]
        # 类别->编号的映射
        self.label_id_dict = label_id_dict
        # 编号->类别的映射
        self.label_id_dict_rev = {
            label_id: label for label, label_id in self.label_id_dict.items()
        }
        return train_data, dev_data, test_data, label_counter

    def samples_balance(self, label_counter, train_data):
        if self.do_sample_balance:
            if self.do_sample_balance == "over":
                std_sample_num = max(label_counter.values())
            else:
                std_sample_num = min(label_counter.values())
            # 计算各个标签的采样比例
            label_sampling_rate_map = {}
            for label, count in label_counter.items():
                sampling_rate = std_sample_num / count
                label_sampling_rate_map[label] = sampling_rate
            balance_samples = []
            for record in train_data:
                label = record["label"]
                sampling_rate = label_sampling_rate_map[label]
                while sampling_rate > 0:
                    if sampling_rate >= 1 or self.guess(sampling_rate):
                        balance_samples.append(record.copy())
                    sampling_rate -= 1
            train_data = balance_samples
        return train_data

    def download_pre_trained_model(self):
        pre_trained_models_config = self.model_lang_weights_dict[
            self.pre_trained_model_type
        ][self.language]
        url = pre_trained_models_config["url"]
        config_path = pre_trained_models_config["config_path"]
        checkpoint_path = pre_trained_models_config["checkpoint_path"]
        dict_path = pre_trained_models_config["dict_path"]

        filename = url.split("/")[-1]
        cache_subdir = file_util.abspath("~/.luwu/cache_models")
        filepath = tf.keras.utils.get_file(
            filename,
            url,
            cache_dir=".",
            cache_subdir=cache_subdir,
            extract=True,
            archive_format="zip",
        )
        # os.remove(filepath)

        cache_subdir = os.path.join(cache_subdir, filename.split(".")[0])
        self.pre_trained_model_config_path = os.path.join(cache_subdir, config_path)
        self.pre_trained_model_checkpoint_path = os.path.join(
            cache_subdir, checkpoint_path
        )
        self.pre_trained_model_dict_path = os.path.join(cache_subdir, dict_path)

    def create_tokenizer(self):
        keep_tokens = []
        if self.simplified_tokenizer:
            token_dict, keep_tokens = load_vocab(
                dict_path=self.pre_trained_model_dict_path,
                simplified=True,
                startswith=["[PAD]", "[UNK]", "[CLS]", "[SEP]"],
            )
            tokenizer = Tokenizer(token_dict, do_lower_case=True)
        else:
            tokenizer = Tokenizer(self.pre_trained_model_dict_path, do_lower_case=True)
        return tokenizer, keep_tokens

    def preprocess_dataset(self):
        """对数据集进行预处理"""

        # 数据集读取和简单处理
        logger.info("读取数据集...")
        train_data, dev_data, test_data, label_counter = self.load_and_split_dataset()
        # 样本均衡
        logger.info("正在进行样本均衡...")
        train_data = self.samples_balance(label_counter, train_data)
        # 混洗数据
        np.random.shuffle(train_data)
        # 下载预训练模型
        logger.info("开始下载预训练模型...")
        self.download_pre_trained_model()
        # 创建tokenizer
        self.tokenizer, self.keep_tokens = self.create_tokenizer()

        logger.info("创建数据生成器...")
        # 创建数据生成器
        if train_data:
            self.train_dataset = TransformerTextClassificationDataGenerator(
                data=train_data,
                batch_size=self.batch_size,
                tokenizer=self.tokenizer,
                maxlen=self.maxlen,
                labels_num=len(self.label_id_dict),
            )
        else:
            raise Exception("训练集为空！请确认！")
        if dev_data:
            self.dev_dataset = TransformerTextClassificationDataGenerator(
                data=dev_data,
                batch_size=self.batch_size,
                tokenizer=self.tokenizer,
                maxlen=self.maxlen,
                labels_num=len(self.label_id_dict),
            )
        else:
            self.dev_dataseet = None
        if test_data:
            self.test_dataset = TransformerTextClassificationDataGenerator(
                data=test_data,
                batch_size=self.batch_size,
                tokenizer=self.tokenizer,
                maxlen=self.maxlen,
                labels_num=len(self.label_id_dict),
            )
        else:
            self.test_dataset = None
        logger.info(
            f"数据集分配：train {len(train_data)}, dev {len(dev_data)}, test {len(test_data)}"
        )

    def train(self):
        # callbacks
        if self.dev_dataset:
            checkpoint_monitor = "val_accuracy"
        else:
            checkpoint_monitor = "accuracy"
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            self.model_save_path, monitor=checkpoint_monitor, save_best_only=True
        )
        params = {
            "epochs": self.epochs,
            "steps_per_epoch": len(self.train_dataset),
            "callbacks": [checkpoint],
        }
        if self.dev_dataset:
            params["validation_data"] = self.dev_dataset.for_fit()
            params["validation_steps"] = len(self.dev_dataset)

        # 判断是否需要冻结模型
        if self.frezee_pre_trained_model:
            self.transformer.model.trainable = False

        self.model.fit(self.train_dataset.for_fit(), **params)

        if self.test_dataset:
            logger.info("在测试集上进行验证...")
            evaluate_dataset = self.test_dataset
        elif self.dev_dataset:
            logger.info("在验证集上进行验证...")
            evaluate_dataset = self.dev_dataset
        else:
            logger.info("在训练集上进行验证...")
            evaluate_dataset = self.train_dataset
        logger.info(
            self.model.evaluate(evaluate_dataset.for_fit(), steps=len(evaluate_dataset))
        )

    def get_call_code(self):
        """返回模型定义和模型调用的代码"""
        if not self._call_code:
            template_path = os.path.join(
                os.path.dirname(__file__), "templates/TransformerTextClassifier.jinja"
            )
            with open(template_path, "r") as f:
                text = f.read()
            data = {
                "dict_path": self.pre_trained_model_dict_path,
                "model_path": self.model_save_path,
                "id_label_dict": str(self.label_id_dict_rev),
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

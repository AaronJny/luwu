# -*- coding: utf-8 -*-
# @Author       : AaronJny
# @LastEditTime : 2021-03-16
# @FilePath     : /LuWu/luwu/core/models/complex/od/models.py
# @Desc         :
import os
from glob import glob

import tensorflow as tf
from jinja2 import Template
from loguru import logger
from luwu.core.models.complex.od.utils import label_map_util
from luwu.utils import cmd_util, file_util


class LuWuObjectDetector:
    def __init__(
        self,
        origin_dataset_path: str = "",
        tfrecord_dataset_path: str = "",
        label_map_path: str = "",
        model_save_path: str = "",
        batch_size: int = 8,
        steps: int = 2000,
        project_id: int = 0,
        **kwargs,
    ):
        """陆吾目标检测模型基类
        # TODO:暂时不增加验证集相关功能，后面再加

        Args:
            origin_dataset_path (str): 处理前的数据集路径
            tfrecord_dataset_path (str): 处理后的tfrecord数据集路径
            label_map_path (str): 目标检测类别映射表(pbtxt)
            model_save_path (str): 模型保存路径
            batch_size (int): mini batch 大小
            steps (int, optional): 训练steps数量. Defaults to 2000.
            project_id (int, optional): 项目编号. Defaults to 0.
        """
        self._call_code = ""
        self.project_id = project_id
        origin_dataset_path = file_util.abspath(origin_dataset_path)
        tfrecord_dataset_path = file_util.abspath(tfrecord_dataset_path)
        label_map_path = file_util.abspath(label_map_path)
        model_save_path = file_util.abspath(model_save_path)
        self.origin_dataset_path = origin_dataset_path
        # 当未给定处理后数据集的路径时，默认保存到原始数据集相同路径
        if tfrecord_dataset_path:
            # 区分给定的是文件夹还是文件。
            # 如果是文件夹，则需要生成tfrecord文件
            # 如果指定指定到文件，则直接使用指定文件，跳过生成步骤
            if os.path.isfile(tfrecord_dataset_path):
                self.tfrecord_dataset_file_path = tfrecord_dataset_path
                self.tfrecord_dataset_dir = os.path.dirname(tfrecord_dataset_path)
                self.need_generate_tfrecord = False
            else:
                self.tfrecord_dataset_dir = tfrecord_dataset_path
                self.tfrecord_dataset_file_path = os.path.join(
                    self.tfrecord_dataset_dir, "train.tfrecord"
                )
                self.need_generate_tfrecord = True
        else:
            self.tfrecord_dataset_dir = self.origin_dataset_path
            self.tfrecord_dataset_file_path = os.path.join(
                self.tfrecord_dataset_dir, "train.tfrecord"
            )
            self.need_generate_tfrecord = True
        # 当未给定pbtxt路径时，也默认保存到tfrecord相同目录下
        if label_map_path:
            if os.path.isfile(label_map_path):
                self.label_map_file_path = label_map_path
                self.label_map_dir = os.path.dirname(self.label_map_file_path)
                self.need_generate_label_map = False
            else:
                self.label_map_dir = label_map_path
                self.label_map_file_path = os.path.join(
                    self.label_map_dir, "label_map.pbtxt"
                )
                self.need_generate_label_map = True
        else:
            self.label_map_dir = self.tfrecord_dataset_dir
            self.label_map_file_path = os.path.join(
                self.label_map_dir, "label_map.pbtxt"
            )
            self.need_generate_label_map = True
        # 当未给定模型保存路径时，默认保存到处理后数据集相同路径
        if self.project_id:
            self.project_save_name = f"luwu-object-detection-project-{self.project_id}"
        else:
            self.project_save_name = f"luwu-object-detection-project"
        if model_save_path:
            self.project_save_path = os.path.join(
                model_save_path, self.project_save_name
            )
        else:
            self.project_save_path = os.path.join(
                self.tfrecord_dataset_dir, self.project_save_name
            )
        self.batch_size = batch_size
        self.steps = steps

    def preprocess_dataset(self):
        """对数据进行预处理"""
        raise NotImplementedError

    def train(self):
        """训练模型"""
        raise NotImplementedError

    def export_model(self):
        """导出训练好的模型"""
        raise NotImplementedError

    def run(self):
        """执行入口"""
        raise NotImplementedError


class LuWuTFModelsObjectDetector(LuWuObjectDetector):
    """目标检测器。
    基于 [https://github.com/tensorflow/models](https://github.com/tensorflow/models)
    """

    fine_tune_models_config_map = {
        "SSD ResNet50 V1 FPN 640x640 (RetinaNet50)": {
            "url": "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz",
            "speed": "46",
            "coco mAP": "34.3",
            "outputs": "Boxes",
            "template": "SSD_ResNet50_V1_FPN_640x640_(RetinaNet50).jinja",
        },
        "EfficientDet D0 512x512": {
            "url": "http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz",
            "speed": "39",
            "coco mAP": "33.6",
            "outputs": "Boxes",
            "template": "EfficientDet_D0_512x512.jinja",
        },
        "CenterNet Resnet101 V1 FPN 512x512": {
            "url": "http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_resnet101_v1_fpn_512x512_coco17_tpu-8.tar.gz",
            "speed": "34",
            "coco mAP": "34.2",
            "outputs": "Boxes",
            "template": "CenterNet_Resnet101_V1_FPN_512x512.jinja",
        },
        "CenterNet HourGlass104 512x512": {
            "url": "http://download.tensorflow.org/models/object_detection/tf2/20200713/centernet_hg104_512x512_coco17_tpu-8.tar.gz",
            "speed": "70",
            "coco mAP": "41.9",
            "outputs": "Boxes",
            "template": "CenterNet_HourGlass104_512x512.jinja",
        },
    }

    def __init__(
        self,
        origin_dataset_path: str = "",
        tfrecord_dataset_path: str = "",
        label_map_path: str = "",
        do_fine_tune: bool = True,
        fine_tune_checkpoint_path: str = "",
        fine_tune_model_name: str = "",
        model_save_path: str = "",
        batch_size: int = 8,
        steps: int = 2000,
        project_id: int = 0,
        **kwargs,
    ):
        """陆吾目标检测模型基类
        # TODO:暂时不增加验证集相关功能，后面再加

        Args:
            origin_dataset_path (str): 处理前的数据集路径
            tfrecord_dataset_path (str): 处理后的tfrecord数据集路径
            label_map_path (str): 目标检测类别映射表(pbtxt)
            do_fine_tune (bool): 是否在预训练模型的基础上进行微调
            fine_tune_checkpoint_path (str): 预训练权重路径
            fine_tune_model_name (str): 预训练模型名称
            model_save_path (str): 模型保存路径
            batch_size (int): mini batch 大小
            steps (int, optional): 训练steps数量. Defaults to 2000.
            project_id (int, optional): 项目编号. Defaults to 0.
        """
        super().__init__(
            origin_dataset_path=origin_dataset_path,
            tfrecord_dataset_path=tfrecord_dataset_path,
            label_map_path=label_map_path,
            model_save_path=model_save_path,
            batch_size=batch_size,
            steps=steps,
            project_id=project_id,
            **kwargs,
        )
        self.do_fine_tune = do_fine_tune
        fine_tune_checkpoint_path = file_util.abspath(fine_tune_checkpoint_path)
        self.fine_tune_checkpoint_path = fine_tune_checkpoint_path
        self.fine_tune_model_name = fine_tune_model_name
        if self.fine_tune_model_name not in self.fine_tune_models_config_map:
            raise Exception(
                f"暂不支持的 object detection model! {self.fine_tune_model_name}"
            )

    def preprocess_dataset(self):
        """对数据集进行预处理，并定义pipeline.config"""
        # 生成 label_map.pbtxt
        if self.need_generate_label_map:
            logger.info("遍历数据集，生成 label_map.pbtxt ...")
            if not os.path.exists(self.origin_dataset_path):
                raise FileNotFoundError("origin_dataset_path 未指定！")
            label_map_util.create_label_map(
                self.origin_dataset_path, self.label_map_file_path
            )
            logger.info(f"label_map 文件已保存到 {self.label_map_file_path}.")
        else:
            logger.info(f"label_map 文件已存在，路径为{self.label_map_file_path}，跳过！")

        # 生成 tfrecord
        if self.need_generate_tfrecord:
            logger.info("遍历数据集，生成 tfrecord ...")
            python_execute_path = cmd_util.get_python_execute_path()
            script_path = os.path.join(
                os.path.dirname(__file__), "utils", "generate_tfrecord.py"
            )
            csv_path = os.path.join(self.tfrecord_dataset_dir, "tmp.csv")
            cmd = f"{python_execute_path} {script_path} -x {self.origin_dataset_path} -l {self.label_map_file_path} -o {self.tfrecord_dataset_file_path} -c {csv_path}"
            cmd_util.run_cmd(cmd)
            if os.path.exists(csv_path):
                os.remove(csv_path)
        else:
            logger.info(f"tfrecord 文件已存在，路径为{self.tfrecord_dataset_file_path}，跳过！")

        if self.do_fine_tune:
            # 下载预训练权重
            url = self.fine_tune_models_config_map[self.fine_tune_model_name]["url"]
            download_dir_name = url.split("/")[-1].split(".")[0]
            cache_dir_path = os.path.expanduser(
                "~/.luwu/tensorflow-models/object-detection/"
            )
            if self.fine_tune_checkpoint_path:
                # 如果给定路径指向了checkpoint文件夹下ckpt-0.index文件
                if self.fine_tune_checkpoint_path.endswith(".index"):
                    file_path = self.fine_tune_checkpoint_path
                    self.fine_tune_checkpoint_path = (
                        self.fine_tune_checkpoint_path.rstrip(".index")
                    )
                # 指向checkpoint文件夹
                elif (
                    (
                        self.fine_tune_checkpoint_path.endswith("checkpoint/")
                        or self.fine_tune_checkpoint_path.endswith("checkpoint")
                    )
                    and os.path.exists(self.fine_tune_checkpoint_path)
                    and os.path.isdir(self.fine_tune_checkpoint_path)
                ):
                    file_path = os.path.join(
                        self.fine_tune_checkpoint_path, "ckpt-0.index"
                    )
                    self.fine_tune_checkpoint_path = os.path.join(
                        self.fine_tune_checkpoint_path, "ckpt-0"
                    )
                # 默认的checkpoint路径
                else:
                    self.fine_tune_checkpoint_path = os.path.join(
                        cache_dir_path,
                        download_dir_name,
                        "checkpoint/ckpt-0",
                    )
                    file_path = self.fine_tune_checkpoint_path + ".index"
            else:
                self.fine_tune_checkpoint_path = os.path.join(
                    cache_dir_path,
                    download_dir_name,
                    "checkpoint/ckpt-0",
                )
                file_path = self.fine_tune_checkpoint_path + ".index"
            # 检查文件是否存在
            if os.path.exists(file_path):
                logger.info(f"预训练权重已存在 {self.fine_tune_checkpoint_path}，跳过！")
            else:
                tf.keras.utils.get_file(
                    download_dir_name, url, untar=True, cache_subdir=cache_dir_path
                )
                logger.info("预训练权重下载完成！")
        else:
            self.fine_tune_checkpoint_path = ""
            logger.info("不使用预训练权重！")

        # 创建项目文件夹结构
        # 根文件夹，如果存在则会抛出异常
        if os.path.exists(self.project_save_path):
            while True:
                text = input(
                    f"目录 {self.project_save_path} 已存在，请更换目录或者确认清空该文件夹！确认清空？[Y/N]"
                )
                text = text.lower().strip()
                if text == "y":
                    cmd = f"rm -rf {self.project_save_path} -y"
                    cmd_util.run_cmd(cmd)
                    os.makedirs(self.project_save_path)
                    break
                elif text == "n":
                    logger.info("请重选选择模型保存的目录，程序已退出。")
                    exit(-1)
                else:
                    continue
        else:
            os.makedirs(self.project_save_path)
        # 训练checkpoint保存路径
        self.train_checkpoint_path = os.path.join(
            self.project_save_path, "train_models"
        )
        os.makedirs(self.train_checkpoint_path, exist_ok=True)
        # 导出模型（SavedModel）保存路径
        self.export_model_path = os.path.join(self.project_save_path, "exported-models")
        os.makedirs(self.export_model_path, exist_ok=True)

        # 创建pipeline.config
        self.generate_pipeline_config()
        logger.info("pipeline.config 生成完毕！")

    def render_template(self, template_path, params) -> str:
        """读取并使用给定的参数渲染模板，返回渲染后的内容

        Args:
            template_path (str): 模板路径
            params (dict)): 填充模板的参数集

        Return:
            str: 渲染之后的template内容
        """
        with open(template_path, "r") as f:
            text = f.read()
        template = Template(text)
        content = template.render(**params)
        return content

    def generate_pipeline_config(self, *args, **kwargs):
        params = {}
        label_map_dict = label_map_util.get_label_map_dict(self.label_map_file_path)
        num_classes = len(label_map_dict)
        # TODO: 优化初始步数生成规则，不再使用默认值
        warmup_steps = max(10, int(self.steps * 0.01))
        warmup_steps = min(warmup_steps, 3000)
        warmup_steps = min(warmup_steps, self.steps)
        if self.fine_tune_model_name in (
            "SSD ResNet50 V1 FPN 640x640 (RetinaNet50)",
            "EfficientDet D0 512x512",
            "CenterNet Resnet101 V1 FPN 512x512",
            "CenterNet HourGlass104 512x512",
        ):
            params["num_classes"] = num_classes
            params["batch_size"] = self.batch_size
            params["num_steps"] = self.steps
            params["fine_tune_checkpoint"] = self.fine_tune_checkpoint_path
            params["label_map_path"] = self.label_map_file_path
            params["train_input_path"] = self.tfrecord_dataset_file_path
            params["eval_input_path"] = self.tfrecord_dataset_file_path
            params["warmup_steps"] = warmup_steps
        else:
            raise Exception(
                f"暂不支持的 object detection model! {self.fine_tune_model_name}"
            )
        template_name = self.fine_tune_models_config_map[self.fine_tune_model_name][
            "template"
        ]
        template_path = template_path = os.path.join(
            os.path.dirname(__file__), f"templates/pipeline/{template_name}"
        )
        content = self.render_template(template_path, params)
        self.train_pipeline_config_path = os.path.join(
            self.train_checkpoint_path, "pipeline.config"
        )
        with open(self.train_pipeline_config_path, "w") as f:
            f.write(content)

    def train(self):
        logger.info("开始训练...")
        python_execute_path = cmd_util.get_python_execute_path()
        cmd = f"{python_execute_path} -m object_detection.model_main_tf2 --model_dir={self.train_checkpoint_path} --pipeline_config_path={self.train_pipeline_config_path}"
        cmd_util.run_cmd(cmd)
        logger.info("训练完成！")

    def export_model(self):
        """将训练好的模型导出为pb格式"""
        logger.info("正在导出模型...")
        python_execute_path = cmd_util.get_python_execute_path()
        cmd = f"{python_execute_path} -m object_detection.exporter_main_v2 --input_type image_tensor --pipeline_config_path {self.train_pipeline_config_path} --trained_checkpoint_dir {self.train_checkpoint_path} --output_directory {self.export_model_path}"
        cmd_util.run_cmd(cmd)

        # 复制一份label_map.pbtxt到该目录
        logger.info("导出 label_map.pbtxt...")
        target_path = os.path.join(self.export_model_path, "label_map.pbtxt")
        cmd = f"cp {self.label_map_file_path} {target_path}"
        cmd_util.run_cmd(cmd)

        logger.info("导出测试代码...")
        if os.path.exists(self.origin_dataset_path):
            filenames = glob(os.path.join(self.origin_dataset_path, "*.jpg"))
            if len(filenames):
                eval_image_path = filenames[0]
            else:
                eval_image_path = ""
        else:
            eval_image_path = ""
        params = {
            "path_to_saved_model": os.path.join(self.export_model_path, "saved_model"),
            "path_to_label_map": target_path,
            "path_to_eval_image": eval_image_path,
        }
        template_path = template_path = os.path.join(
            os.path.dirname(__file__), "templates/project/eval.jinja"
        )
        content = self.render_template(template_path, params)
        target_path = os.path.join(self.project_save_path, "eval.py")
        with open(target_path, "w") as f:
            f.write(content)
        logger.info("导出完成！")

    def run(self):
        # 预处理
        self.preprocess_dataset()
        # 训练
        self.train()
        # 导出模型
        self.export_model()

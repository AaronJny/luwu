# -*- coding: utf-8 -*-
# @Author       : AaronJny
# @LastEditTime : 2021-05-12
# @FilePath     : /LuWu/luwu/core/models/kaggle/kaggle.py
# @Desc         :
import os
from uuid import uuid1
import subprocess
import json
import re
import time

from loguru import logger
from luwu.core.models.classifier import LuwuImageClassifier
from luwu.core.models.complex.od.models import LuWuTFModelsObjectDetector
from luwu.core.models.text_classifier.transformers import TransformerTextClassification
from luwu.utils import cmd_util, file_util


class KaggleUtil:
    def __init__(self, luwu_model_class, *args, **kwargs):
        self.luwu_model_class = luwu_model_class
        # 是否使用kaggle硬件加速
        if "kaggle_accelerator" not in kwargs:
            kwargs["kaggle_accelerator"] = False
        self.kwargs = kwargs
        self.uuid = str(uuid1())
        self.tmp_dir_path = file_util.get_tmp_dir(dir_name=self.uuid)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.clean_tmp_files()

    def load_notebook_metadata(self):
        file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "templates", "notebook.json"
        )
        with open(file_path, "r") as f:
            metadata = json.load(f)
        return metadata

    def update_notebook_codes(self, metadata: dict, codes: list):
        metadata["cells"][0]["source"] = codes

    def upload_dataset(self):
        origin_dataset_path = self.kwargs.get("origin_dataset_path", "")
        if os.path.exists(origin_dataset_path):
            # 先复制一份数据集
            # 创建一个文件夹
            dataset_path = os.path.join(self.tmp_dir_path, "kaggle-data")
            copy_path = os.path.join(dataset_path, "data")
            logger.info(f"创建文件夹 {copy_path} ...")
            file_util.mkdirs(copy_path)
            # 复制数据集到临时目录
            logger.info(f"复制数据集到临时目录...")
            if os.path.isdir(origin_dataset_path):
                cmd = f'cp -r {os.path.join(origin_dataset_path,"*")} {copy_path}'
            else:
                cmd = f"cp -r {origin_dataset_path} {copy_path}"
            cmd_util.run_cmd(cmd)
            # 使用kaggle api初始化数据集
            logger.info("使用kaggle api初始化数据集...")
            cmd = f"kaggle datasets init -p {dataset_path}"
            cmd_util.run_cmd(cmd)
            # 配置dataset meta
            dataset_meta_path = os.path.join(dataset_path, "dataset-metadata.json")
            with open(dataset_meta_path, "r") as f:
                dataset_meta = json.load(f)
            dataset_meta["title"] = f"luwu-dataset-{self.uuid}"
            dataset_meta["id"] = (
                dataset_meta["id"].split("/")[0] + "/" + f"luwu-dataset-{self.uuid}"
            )
            with open(dataset_meta_path, "w") as f:
                json.dump(dataset_meta, f, ensure_ascii=False, indent=2)
            # 上传数据集
            logger.info("上传数据集...")
            cmd = f"kaggle datasets create -r zip -p {dataset_path}"
            cmd_util.run_cmd(cmd)
            logger.info("数据集上传完成！")
            logger.info("等待 kaggle 处理数据集，这可能需要几分钟时间 ...")
            self.dataset_id = dataset_meta["id"]
            self.dataset_title = dataset_meta["title"]
            cmd = f"kaggle datasets status {self.dataset_id}"
            while True:
                code, output = subprocess.getstatusoutput(cmd)
                if code != 0:
                    logger.error(output)
                    raise Exception("查询数据集状态失败！")
                if output:
                    if "ready" in output:
                        logger.info("数据集准备完成！")
                    else:
                        logger.warning(output)
                    break
                else:
                    logger.info("暂未查询到数据，等待中 ...")
                    time.sleep(10)
        else:
            raise FileNotFoundError(
                f"指定的 origin_dataset_path 不存在！{origin_dataset_path}"
            )

    def train_on_kaggle(self, task_type):
        # 生成训练代码
        # 创建文件夹
        kernel_path = os.path.join(self.tmp_dir_path, "kaggle-kernel")
        logger.info(f"创建文件夹 {kernel_path} ...")
        file_util.mkdirs(kernel_path)
        # 初始化kernel
        logger.info("使用 kaggle api 初始化 kernel ...")
        cmd = f"kaggle kernels init -p {kernel_path}"
        cmd_util.run_cmd(cmd)
        # 生成训练脚本
        override_params = {"project_id", "cmd", "luwu_version"}
        train_cmd_params = []
        if task_type == "classification":
            project_name = "luwu-classification-project"
            override_params.update(["net_name", "network_name"])
            # tfrecord数据集路径
            tfrecord_dataset_path = "./dataset"
            train_cmd_params.append(f"--tfrecord_dataset_path {tfrecord_dataset_path}")
            override_params.add("tfrecord_dataset_path")
        elif task_type == "detection":
            project_name = "luwu-object-detection-project"
            override_params.update(
                [
                    "label_map_path",
                    "fine_tune_checkpoint_path",
                ]
            )
            # tfrecord数据集路径
            tfrecord_dataset_path = "./dataset"
            train_cmd_params.append(f"--tfrecord_dataset_path {tfrecord_dataset_path}")
            override_params.add("tfrecord_dataset_path")
        elif task_type == "text_classification":
            project_name = "luwu-text-classification-project"
        else:
            raise Exception(f"不支持的任务类型! {task_type}")

        # 原始数据集路径
        origin_dataset_path = os.path.join("../input", self.dataset_title)
        if self.kwargs.get("cmd") == "text_classification":
            filename = self.kwargs.get("origin_dataset_path").split("/")[-1]
            origin_dataset_path = os.path.join(origin_dataset_path, filename)
        train_cmd_params.append(f"--origin_dataset_path {origin_dataset_path}")
        override_params.add("origin_dataset_path")
        # 模型保存路径
        model_save_path = "./project"
        train_cmd_params.append(f"--model_save_path {model_save_path}")
        override_params.add("model_save_path")
        # 其他参数
        for arg_name, arg_value in self.kwargs.items():
            if "kaggle" in arg_name:
                continue
            if arg_name in override_params:
                continue
            # 兼容bool类型参数
            if arg_value != False:
                train_cmd_params.append(f'--{arg_name} "{arg_value}"')
            # else:
            #     train_cmd_params.append(f"--{arg_name}")
        if task_type == "classification":
            train_cmd = f"!luwu {task_type} {' '.join(train_cmd_params)} {self.luwu_model_class.__name__}\n"
        elif task_type == "detection":
            train_cmd = f"!luwu {task_type} {' '.join(train_cmd_params)}\n"
        elif task_type == "text_classification":
            train_cmd = f"!luwu {task_type} {' '.join(train_cmd_params)}\n"
        else:
            raise Exception(f"不支持的任务类型! {task_type}")
        project_path = os.path.join(model_save_path, project_name)
        if task_type == "classification":
            zip_cmd = (
                f"!mv {project_path} ./ "
                f"&& zip -r {project_name}-{self.uuid}.zip ./{project_name} "
                f"&& rm -rf {tfrecord_dataset_path} "
                f"&& rm -rf ./{project_name} "
                f"&& rm -rf {model_save_path} \n"
            )
        elif task_type == "detection":
            zip_cmd = (
                f"!mv {project_path} ./ "
                f'&& rm -rf {os.path.join(project_name,"train_models")} '
                f"&& zip -r {project_name}-{self.uuid}.zip ./{project_name} "
                f"&& rm -rf {tfrecord_dataset_path} "
                f"&& rm -rf ./{project_name} "
                f"&& rm -rf {model_save_path} \n"
            )
        elif task_type == "text_classification":
            zip_cmd = (
                f"!mv {project_path} ./ "
                f"&& zip -r {project_name}-{self.uuid}.zip ./{project_name} "
                f"&& rm -rf ./{project_name} "
                f"&& rm -rf {model_save_path} \n"
            )
        luwu_version = self.kwargs.get("luwu_version")
        if luwu_version:
            install_cmd = f"!pip install luwu=={luwu_version}\n"
        else:
            install_cmd = "!pip install luwu\n"
        codes = [
            "# 安装 luwu\n",
            install_cmd,
            "# 执行训练指令\n",
            train_cmd,
            "# 打包待下载文件的指令\n",
            zip_cmd,
            "    ",
        ]
        script_metadata = self.load_notebook_metadata()
        self.update_notebook_codes(script_metadata, codes)
        kernel_file_path = os.path.join(kernel_path, f"luwu-kernel-{self.uuid}.ipynb")
        with open(kernel_file_path, "w") as f:
            json.dump(script_metadata, f, ensure_ascii=False, indent=2)
        # 修改 kernel-metadata.json
        kernel_metadata_path = os.path.join(kernel_path, "kernel-metadata.json")
        with open(kernel_metadata_path, "r") as f:
            kernel_metadata = json.load(f)
        kernel_metadata["id"] = (
            kernel_metadata["id"].split("/")[0] + "/" + f"luwu-kernel-{self.uuid}"
        )
        kernel_metadata["title"] = f"luwu-kernel-{self.uuid}"
        kernel_metadata["code_file"] = kernel_file_path
        kernel_metadata["language"] = "python"
        kernel_metadata["kernel_type"] = "notebook"
        kaggle_accelerator = self.kwargs.get("kaggle_accelerator", False)
        if kaggle_accelerator:
            kernel_metadata["enable_gpu"] = "true"
        else:
            kernel_metadata["enable_gpu"] = "false"
        kernel_metadata["dataset_sources"] = [
            self.dataset_id,
        ]
        with open(kernel_metadata_path, "w") as f:
            json.dump(kernel_metadata, f, ensure_ascii=False, indent=2)
        logger.info(f"kernel metadata :{kernel_metadata}")
        self.kernel_id = kernel_metadata["id"]
        self.kernel_title = kernel_metadata["title"]
        # 推送并运行kernel
        logger.info("将 kernel 推送到 Kaggle 并运行 ...")
        cmd = f"kaggle kernels push -p {kernel_path}"
        logger.debug(cmd)
        cmd_util.run_cmd(cmd)
        logger.info("推送完成！等待运行中 ...")
        running = False
        error_cnt = 0
        while True:
            cmd = f"kaggle kernels status {self.kernel_id}"
            code, output = subprocess.getstatusoutput(cmd)
            if code != 0:
                logger.error(output)
                raise Exception(output)
            pattern = 'has status "([^"]*)"'
            matches = re.findall(pattern, output)
            if not matches:
                logger.error(f"未查询到状态！{output}")
                error_cnt += 1
                if error_cnt > 10:
                    raise Exception(f"连续10次未获取到 kernel {self.kernel_id} 的运行状态！")
            else:
                status = matches[0]
                # 运行之前，所有的状态都忽略
                if not running:
                    if status == "running":
                        logger.info(f"{self.kernel_id} running ...")
                        running = True
                else:
                    # 运行之后，找到第一次非 running 状态就退出
                    if status == "running":
                        logger.info(f"{self.kernel_id} running ...")
                    else:
                        self.kernel_exit_status = status
                        logger.info(output)
                        logger.info(
                            f"{self.kernel_id} 终止状态：{self.kernel_exit_status} . 已退出！"
                        )
                        break
                time.sleep(10)
        logger.info("kernel 运行已结束！")

    def download_result_from_kaggle(self):
        output_path = os.path.join(self.tmp_dir_path, "kaggle-output")
        logger.info(f"创建文件夹 {output_path} ...")
        file_util.mkdirs(output_path)
        logger.info("从kaggle拉取运行结果...")
        cmd = f"kaggle kernels output {self.kernel_id} -p {output_path}"
        cmd_util.run_cmd(cmd)
        model_save_path = self.kwargs.get("model_save_path", "")
        if not model_save_path:
            model_save_path = "luwu-output"
        project_path = file_util.abspath(model_save_path)
        file_util.mkdirs(project_path)
        output_files_path = os.path.join(output_path, "*")
        logger.info(f"将运行结果移动到指定目录 {project_path} ...")
        cmd = f"cp -r {output_files_path} {project_path}"
        cmd_util.run_cmd(cmd)
        logger.info("Done.")

    def clean_tmp_files(self):
        """删除过程中生成的临时文件（本地的）"""
        cmd = f"rm -rf {self.tmp_dir_path}"
        cmd_util.run_cmd(cmd)
        tmp_dir_parent = "/".join(self.tmp_dir_path.split("/")[:-1])
        # 确保安全删除
        if ".luwu/tmp/" in tmp_dir_parent:
            cmd = f"rm -rf {tmp_dir_parent}"
            cmd_util.run_cmd(cmd)

    def image_classification_entry(self):
        # 处理原始数据
        logger.info("[*][1]开始处理数据集...")
        self.upload_dataset()
        # 使用Kaggle训练
        logger.info("[*][2]开始处理kernel...")
        self.train_on_kaggle(task_type=self.kwargs.get("cmd"))
        # 下载训练好的数据
        logger.info("[*][3]开始拉取运行结果...")
        self.download_result_from_kaggle()
        # 清理临时文件
        logger.info("[*][4]清理临时文件...")
        self.clean_tmp_files()
        logger.info("[*]完成！")

    def text_classification_entry(self):
        # 处理原始数据
        logger.info("[*][1]开始处理数据集...")
        self.upload_dataset()
        # 使用Kaggle训练
        logger.info("[*][2]开始处理kernel...")
        self.train_on_kaggle(task_type=self.kwargs.get("cmd"))
        # 下载训练好的数据
        logger.info("[*][3]开始拉取运行结果...")
        self.download_result_from_kaggle()
        # 清理临时文件
        logger.info("[*][4]清理临时文件...")
        self.clean_tmp_files()
        logger.info("[*]完成！")

    def object_detection_entry(self):
        # 处理原始数据
        logger.info("[*][1]开始处理数据集...")
        self.upload_dataset()
        # 使用Kaggle训练
        logger.info("[*][2]开始处理kernel...")
        self.train_on_kaggle(task_type=self.kwargs.get("cmd"))
        # 下载训练好的数据
        logger.info("[*][3]开始拉取运行结果...")
        self.download_result_from_kaggle()
        # 清理临时文件
        logger.info("[*][4]清理临时文件...")
        self.clean_tmp_files()
        logger.info("[*]完成！")

    def run(self):
        # 分类任务
        if issubclass(self.luwu_model_class, LuwuImageClassifier):
            self.image_classification_entry()
        # 目标检测任务
        elif issubclass(self.luwu_model_class, LuWuTFModelsObjectDetector):
            self.object_detection_entry()
        elif issubclass(self.luwu_model_class, TransformerTextClassification):
            self.text_classification_entry()
        else:
            raise Exception(f"不支持的任务类型！{self.luwu_model_class}")

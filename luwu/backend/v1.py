# -*- coding: utf-8 -*-
# @Author       : AaronJny
# @LastEditTime : 2021-01-23
# @FilePath     : /app/luwu/backend/v1.py
# @Desc         :
from flask.blueprints import Blueprint
from luwu.backend import status_code_wrapper
from flask import request
import os

api_v1_blueprint = Blueprint("api_v1_blueprint", __name__, url_prefix="/api/v1")

ENGINE_LIST = [
    {"index": 1, "name": "预设模型", "tip": ""},
    # {"index": 2, "name": "AutoKeras", "tip": ""},
    # {"index": 3, "name": "KerasTuner", "tip": ""},
    # {"index": 4, "name": "NNI", "tip": ""},
]


def check_path_correct(path):
    """检查给定路径是否符合要求，不符合要求则会抛出异常

    Args:
        path (str): 待检查路径
    """
    if not os.path.exists(path):
        raise Exception(f"指定路径 {path} 不存在！")
    if not os.path.isdir(path):
        raise Exception(f"{path} 必须是文件夹！")


@api_v1_blueprint.route("/")
def index():
    return "hello world"


@api_v1_blueprint.route("/image/classifier/engines/")
@status_code_wrapper()
def get_engine_list():
    data = ENGINE_LIST
    return data


@api_v1_blueprint.route("/image/classifier/project/create/", methods=["POST"])
@status_code_wrapper()
def create_image_classify_project():
    # 模型引擎
    engine_index = request.json.get("engine_index")
    index_engine_map = {item["index"]: item["name"] for item in ENGINE_LIST}
    engine = index_engine_map.get(engine_index, "")
    if not engine:
        raise Exception("指定模型引擎不存在！")
    # 原始数据及
    dataset_index = request.json.get("dataset_index")
    origin_dataset_path = request.json.get("origin_dataset_path", "")
    check_path_correct(origin_dataset_path)
    # 清洗后的数据集保存路径
    target_dataset_path = request.json.get("target_dataset_path", "")
    if target_dataset_path:
        check_path_correct(target_dataset_path)
    else:
        target_dataset_path = origin_dataset_path
    # 模型保存路径
    model_save_path = request.json.get("model_save_path", "")
    if model_save_path:
        check_path_correct(model_save_path)
    else:
        model_save_path = target_dataset_path
    # 训练参数
    batch_size = int(request.json.get("batch_size", 32))
    epochs = int(request.json.get("epochs", 30))
    if dataset_index == 1:
        pass
    else:
        raise Exception("不支持的数据集类型！")
    print(
        engine,
        dataset_index,
        origin_dataset_path,
        target_dataset_path,
        model_save_path,
        batch_size,
        epochs,
    )

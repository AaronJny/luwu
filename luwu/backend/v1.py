# -*- coding: utf-8 -*-
# @Author       : AaronJny
# @LastEditTime : 2021-03-16
# @FilePath     : /LuWu/luwu/backend/v1.py
# @Desc         :
import json
import os
import re
import time

from flask import request
from flask.blueprints import Blueprint

from luwu.backend import status_code_wrapper
from luwu.backend.model import TrainProject, db
from luwu.core.models import image

api_v1_blueprint = Blueprint("api_v1_blueprint", __name__, url_prefix="/api/v1")

ENGINE_LIST = [
    {"index": 1, "name": "预设模型", "tip": "预先定义好的、结构固定的模型"},
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


@api_v1_blueprint.route("/image/classifier/models/<index>/")
@status_code_wrapper()
def get_image_classifier_list(index):
    index_engine_map = {item["index"]: item["name"] for item in ENGINE_LIST}
    engine = index_engine_map.get(int(index), "")
    if not engine:
        raise Exception("指定模型引擎不存在！")
    data = []
    if engine == "预设模型":
        classifiers = list(image.pre_trained_classifiers)
        data = []
        pattern = "Luwu(.*)ImageClassifier"
        for item in classifiers:
            cls_name = item.__name__
            model_name = re.findall(pattern, cls_name)[0]
            data.append({"index": cls_name, "name": model_name, "tip": ""})
    return data


@api_v1_blueprint.route("/image/classifier/project/create/", methods=["POST"])
@status_code_wrapper()
def create_image_classify_project():
    # 模型名称
    model_name = request.json.get("model_name", "")
    if not model_name:
        raise Exception("必须选择一个Model!")
    try:
        model = getattr(image, model_name)
    except:
        raise Exception("选择的模型不存在！")
    # 原始数据集
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
    train_project = TrainProject()
    train_project.params = {
        "dataset_index": dataset_index,
        "origin_dataset_path": origin_dataset_path,
        "tfrecord_dataset_path": target_dataset_path,
        "model_save_path": model_save_path,
        "batch_size": batch_size,
        "epochs": epochs,
    }
    train_project.model_name = str(model)
    train_project.status = 0
    train_project.addtime = int(time.time())
    train_project.add()


@api_v1_blueprint.route("/project/list/")
@status_code_wrapper()
def get_train_project_list():
    projects = db.session.query(TrainProject).all()
    data = [item.to_dict() for item in projects]
    return data


@api_v1_blueprint.route("/project/list/exists/")
@status_code_wrapper()
def get_train_project_exists_list():
    projects = db.session.query(TrainProject).filter(TrainProject.deleted == 0).all()
    data = [item.to_dict() for item in projects]
    return data


@api_v1_blueprint.route("/project/<xid>/")
@status_code_wrapper()
def get_train_project_by_id(xid):
    tp = db.session.query(TrainProject).get(int(xid))
    if not tp:
        raise Exception("指定项目编号不存在！")
    data = tp.to_dict()
    return data


@api_v1_blueprint.route("/project/<xid>/delete/physical/")
@status_code_wrapper()
def physical_delete_train_project_by_id(xid):
    """物理删除指定训练项目"""
    db.session.query(TrainProject).filter(TrainProject.id == int(xid)).delete()
    db.session.commit()


@api_v1_blueprint.route("/project/<xid>/delete/logical/")
@status_code_wrapper()
def logical_delete_train_project_by_id(xid):
    """逻辑删除指定训练项目"""
    tp = db.session.query(TrainProject).get(int(xid))
    if not tp:
        raise Exception("指定项目编号不存在！")
    tp.deleted = 1
    tp.status = 5
    db.session.commit()


@api_v1_blueprint.route("/project/<xid>/status/update/", methods=["POST"])
@status_code_wrapper()
def update_train_project_status(xid):
    tp = db.session.query(TrainProject).get(int(xid))
    if not tp:
        raise Exception("指定项目编号不存在！")
    status = request.json.get("status", None)
    if status is None:
        raise Exception("指定状态不正确！")
    tp.status = status
    db.session.commit()


@api_v1_blueprint.route("/project/<xid>/code/update/", methods=["POST"])
@status_code_wrapper()
def update_train_project_code(xid):
    tp = db.session.query(TrainProject).get(int(xid))
    if not tp:
        raise Exception("指定项目编号不存在！")
    code = request.json.get("code", None)
    if code is None:
        raise Exception("指定状态不正确！")
    tp.code = code
    db.session.commit()


@api_v1_blueprint.route("/project/<xid>/update/", methods=["POST"])
@status_code_wrapper()
def update_train_project_attr(xid):
    tp = db.session.query(TrainProject).get(int(xid))
    if not tp:
        raise Exception("指定项目编号不存在！")
    for attr, value in request.json.items():
        setattr(tp, attr, value)
    db.session.commit()


@api_v1_blueprint.route("/project/<int:xid>/logs/", defaults={"start_line": 0})
@api_v1_blueprint.route("/project/<int:xid>/logs/<int:start_line>/")
@status_code_wrapper()
def read_project_logs_by_id(xid, start_line):
    tp = db.session.query(TrainProject).get(int(xid))
    if not tp:
        raise Exception("指定项目编号不存在！")
    model_save_path = tp.params["model_save_path"]
    log_path = os.path.join(
        model_save_path, f"luwu-classification-project-{xid}", f"train.log"
    )
    if not os.path.exists(log_path):
        raise Exception("暂无日志！请稍后再试...")
    lines = []
    with open(log_path, "r") as f:
        for index, line in enumerate(f):
            if index < start_line:
                continue
            if line.endswith("\n"):
                lines.append(line)
            else:
                break
    current_line = start_line + len(lines) - 1
    text = "<br/>".join(lines)
    if start_line > 0:
        text = "<br/>" + text
    data = {"current_line": current_line, "text": text}
    return data

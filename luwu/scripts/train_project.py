# -*- coding: utf-8 -*-
# @Author       : AaronJny
# @LastEditTime : 2021-01-28
# @FilePath     : /LuWu/luwu/scripts/train_project.py
# @Desc         :
import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), ".."), ".."))
)
import tensorflow as tf

# 引入顺序不能变动，必须先执行此段代码，才能引入luwu下的模块
gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


import fire
from loguru import logger
from luwu.scripts.utils import (
    get_project_by_id,
    load_model_class,
    update_project_code,
    update_project_status,
)


def train(project):
    update_project_status(project["id"], 2)
    model_name = project["model_name"]
    model_class = load_model_class(model_name)
    model = model_class(project_id=project["id"], **project["params"])
    model.run()
    code = model.get_call_code()
    update_project_code(project["id"], code)


@logger.catch
def run(project_id):
    project = get_project_by_id(project_id)
    try:
        train(project)
        update_project_status(project["id"], 2)
        logger.info("处理完成！")
    except Exception as e:
        update_project_status(project["id"], 6)
        logger.error("处理异常！")
        raise e


if __name__ == "__main__":
    fire.Fire(run)

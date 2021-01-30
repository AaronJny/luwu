# -*- coding: utf-8 -*-
# @Author       : AaronJny
# @LastEditTime : 2021-01-30
# @FilePath     : /LuWu/luwu/backend/model.py
# @Desc         :
import json
from datetime import datetime

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class TrainProject(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    params_text = db.Column(db.Text, nullable=False, default="", comment="参数json")
    code = db.Column(db.Text, nullable=False, default="", comment="生成的调用代码")
    model_name = db.Column(db.String(100), nullable=False, default="", comment="模型名称")
    status = db.Column(
        db.Integer,
        nullable=False,
        default=0,
        comment="处理状态 0-未处理，1-等待处理，2-处理中，3-处理完成，4-处理失败，5-手动终止，6-异常退出",
    )
    deleted = db.Column(db.Integer, nullable=False, default=0, comment="是否删除，1-删除")
    addtime = db.Column(db.Integer, nullable=False, default=0, comment="创建时间")

    @property
    def status_text(self):
        status_dict = {
            0: "未处理",
            1: "等待处理",
            2: "处理中",
            3: "处理完成",
            4: "处理失败",
            5: "手动终止",
            6: "异常退出",
        }
        return status_dict[self.status]

    @property
    def params(self):
        return json.loads(self.params_text)

    @params.setter
    def params(self, value):
        self.params_text = json.dumps(value)

    def to_dict(self):
        data = {
            "id": self.id,
            "params": self.params,
            "params_text": self.params_text,
            "code": self.code,
            "model_name": self.model_name,
            "deleted": self.deleted,
            "status": self.status,
            "status_text": self.status_text,
            "addtime": datetime.fromtimestamp(self.addtime).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        }
        return data

    def add(self):
        db.session.add(self)
        db.session.commit()

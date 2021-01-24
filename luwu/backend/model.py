# -*- coding: utf-8 -*-
# @Author       : AaronJny
# @LastEditTime : 2021-01-24
# @FilePath     : /app/luwu/backend/model.py
# @Desc         :
from flask_sqlalchemy import SQLAlchemy
import json
from datetime import datetime

db = SQLAlchemy()


class TrainProject(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    params = db.Column(db.Text, nullable=False, default="", comment="参数json")
    model_name = db.Column(db.String(100), nullable=False, default="", comment="模型名称")
    status = db.Column(
        db.Integer, nullable=False, default=0, comment="处理状态 0-未处理，1-处理中，2-处理完成，3-处理失败"
    )
    addtime = db.Column(db.Integer, nullable=False, default=0, comment="创建时间")

    def to_dict(self):
        data = {
            "id": self.id,
            "params": json.loads(self.params),
            "model_name": self.model_name,
            "status": self.status,
            "addtime": datetime.fromtimestamp(self.addtime).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        }
        return data

    def add(self):
        db.session.add(self)
        db.session.commit()
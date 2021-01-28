# -*- coding: utf-8 -*-
# @Author       : AaronJny
# @LastEditTime : 2021-01-28
# @FilePath     : /app/luwu/backend/config.py
# @Desc         :
import os


class Config:

    LUWU_DIR = os.path.expanduser("~/.luwu")

    SQLALCHEMY_DATABASE_URI = f"sqlite:///{LUWU_DIR}/luwu.db"

    PORT = 7788

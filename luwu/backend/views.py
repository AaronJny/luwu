# -*- coding: utf-8 -*-
# @Author       : AaronJny
# @LastEditTime : 2021-01-29
# @FilePath     : /LuWu/luwu/backend/views.py
# @Desc         :
from flask import Blueprint
from flask import render_template

main_blueprint = Blueprint("main_blueprint", __name__)


@main_blueprint.route("/", defaults={"path": ""})
@main_blueprint.route("/home/<path:path>")
def catch_all(path):
    return render_template("index.html")
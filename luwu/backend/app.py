# -*- coding: utf-8 -*-
# @Author       : AaronJny
# @LastEditTime : 2021-01-28
# @FilePath     : /LuWu/luwu/backend/app.py
# @Desc         :
from flask import Flask
from flask import render_template
from flask_cors import CORS
from werkzeug.routing import BaseConverter

from luwu.backend.config import Config
from luwu.backend.model import db


class RegexConverter(BaseConverter):
    def __init__(self, map, *args):
        self.map = map
        self.regex = args[0]


def create_app():
    app = Flask(
        __name__, static_folder="../dist", template_folder="../dist", static_url_path=""
    )
    app.config.from_object(Config)
    cors = CORS()

    cors.init_app(app)
    db.init_app(app)

    with app.app_context():
        # 初始化数据库
        db.create_all()

    app.url_map.converters["regex"] = RegexConverter

    @app.route("/", defaults={"path": ""})
    @app.route("/home/<path:path>")
    def catch_all(path):
        return render_template("index.html")

    from luwu.backend.v1 import api_v1_blueprint

    app.register_blueprint(api_v1_blueprint)

    from luwu.backend.views import main_blueprint

    app.register_blueprint(main_blueprint)

    return app


def run():
    app = create_app()
    app.run(host="0.0.0.0", port=Config.PORT)


if __name__ == "__main__":
    run()

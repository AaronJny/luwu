# -*- coding: utf-8 -*-
# @Author       : AaronJny
# @LastEditTime : 2021-03-09
# @FilePath     : /LuWu/install_script.py
# @Desc         : Luwu一键安装脚本
import os
import platform
import subprocess
import sys


def stage_tip(num, msg, logger=None):
    total = 7
    message = "Step {}/{}: {}".format(num, total, msg)
    if logger is None:
        print(message)
    else:
        logger.info(message)


def run_cmd(cmd):
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stderr=sys.stderr,
        close_fds=True,
        stdout=sys.stdout,
        universal_newlines=True,
        shell=True,
        bufsize=1,
    )

    proc.communicate()
    code = proc.returncode
    if code != 0:
        exit(code)


def run_install():
    stage_tip(1, "检查系统类型是否支持...")
    system = platform.system()
    if system == "Darwin":
        print("System: MacOS")
        protoc_filename = "protoc-3.15.3-osx-x86_64.zip"
    elif system == "Linux":
        print("System: Linux")
        protoc_filename = "protoc-3.15.3-linux-x86_64.zip"
    elif system == "Windows":
        print("抱歉，LuWu 暂未支持在 Windows 上运行！")
        return -1
    else:
        print("不支持的系统类型 {} ！".format(system))
        return -1

    python_execute_path = sys.executable
    pip_execute_path = f"{python_execute_path} -m pip"

    stage_tip(2, "正在安装 requirements.txt ...")
    cmd = "{} install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ ".format(
        pip_execute_path
    )
    run_cmd(cmd)

    from loguru import logger

    stage_tip(3, "正在下载 TensorFlow Object Detection API ...", logger)
    cmd = "mkdir -p addons && cd addons && \
        rm -rf master* && \
        rm -rf models* && \
        wget https://codeload.github.com/tensorflow/models/zip/master &&\
        unzip master && mv models-master models && rm -rf master*"
    run_cmd(cmd)

    stage_tip(4, "正在下载并安装 protoc ...", logger)
    cmd = "cd addons && \
        mkdir -p protoc && \
        cd protoc && \
        rm -rf * && \
        rm -rf {protoc_filename}* && \
        wget https://github.com/protocolbuffers/protobuf/releases/download/v3.15.3/{protoc_filename} && \
        unzip {protoc_filename} && \
        rm -rf {protoc_filename}* && \
        export PATH=$PATH:`pwd`/bin && \
        cd ../models/research/ && \
        protoc object_detection/protos/*.proto --python_out=.".format(
        protoc_filename=protoc_filename
    )
    run_cmd(cmd)

    stage_tip(5, "正在安装 cocoapi ...", logger)
    cmd = "{} install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI".format(
        pip_execute_path
    )
    run_cmd(cmd)

    stage_tip(6, "正在安装 TensorFlow Object Detection API ...", logger)
    cmd = "cd addons/models/research/ && \
        cp object_detection/packages/tf2/setup.py . && \
        {} -m pip install .".format(
        python_execute_path
    )
    run_cmd(cmd)

    stage_tip(7, "测试是否安装成功 ...", logger)
    cmd = "cd addons/models/research/ && \
        {} object_detection/builders/model_builder_tf2_test.py".format(
        python_execute_path
    )
    run_cmd(cmd)


if __name__ == "__main__":
    run_install()

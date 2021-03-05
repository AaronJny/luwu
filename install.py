# -*- coding: utf-8 -*-
# @Author       : AaronJny
# @LastEditTime : 2021-03-05
# @FilePath     : /LuWu/install.py
# @Desc         : Luwu一键安装脚本
import argparse
import os
import platform
import subprocess
import sys

parser = argparse.ArgumentParser(description="LuWu installation script.")
parser.add_argument(
    "-py",
    "--python_name",
    help="The name of the python shortcut pointing to the Python3.+ environment to be used. The default is python. ",
    type=str,
    default="python",
)
parser.add_argument(
    "-pip",
    "--pip_name",
    help="The name of the pip shortcut pointing to the Python3.+ environment to be used. The default is pip.",
    type=str,
    default="pip",
)
parser.add_argument(
    "--proxy",
    help="Http proxy to increase download speed. e.g: http://localhost:7890",
    type=str,
    default="",
)

args = parser.parse_args()

if args.proxy:
    os.environ["http_proxy"] = args.proxy
    os.environ["https_proxy"] = args.proxy


def stage_tip(num, msg, logger=None):
    total = 8
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


def run():
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

    python_name = args.python_name
    pip_name = args.pip_name
    print("Python Name: {}".format(python_name))
    print("Pip Name: {}".format(pip_name))

    stage_tip(2, "正在安装 requirements.txt ...")
    cmd = "{} install -r requirements.txt".format(pip_name)
    run_cmd(cmd)

    from loguru import logger
    from tqdm import tqdm

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
        pip_name
    )
    run_cmd(cmd)

    stage_tip(6, "正在安装 TensorFlow Object Detection API ...", logger)
    cmd = "cd addons/models/research/ && \
        cp object_detection/packages/tf2/setup.py . && \
        {} -m pip install .".format(
        python_name
    )
    run_cmd(cmd)

    stage_tip(7, "测试是否安装成功 ...", logger)
    cmd = "cd addons/models/research/ && \
        {} object_detection/builders/model_builder_tf2_test.py".format(
        python_name
    )
    run_cmd(cmd)


if __name__ == "__main__":
    run()

# -*- coding: utf-8 -*-
# @Author       : AaronJny
# @LastEditTime : 2021-03-09
# @FilePath     : /LuWu/luwu/utils/cmd_util.py
# @Desc         :
import sys
import os
import subprocess
import shlex


def run_cmd(cmd, raise_exception=True):
    """使用shell执行指定命令，并将执行结果实时输出，返回命令退出的状态码

    Args:
        cmd (str): 待执行的shell命令

    Returns:
        int: shell退出时返回的状态码
    """
    if "JPY_PARENT_PID" in os.environ:
        # code, output = subprocess.getstatusoutput(cmd)
        p = subprocess.Popen(
            shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        while True:
            output = p.stdout.read1(1024).decode("utf-8")
            print(output, end="")
            if p.poll() is not None:
                break
        code = p.returncode
    else:
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
    if raise_exception and code != 0:
        raise Exception("命令运行过程中出错！")
    return code


def get_python_execute_path():
    """获取luwu安装到的python名称

    Returns:
        [type]: [description]
    """
    python_execute_path = sys.executable
    return python_execute_path
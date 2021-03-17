# -*- coding: utf-8 -*-
# @Author       : AaronJny
# @LastEditTime : 2021-03-16
# @FilePath     : /LuWu/luwu/scripts/scheduler.py
# @Desc         :
import multiprocessing
import os
import subprocess
import time

from loguru import logger
from luwu.scripts.utils import (
    delete_project_by_id,
    read_train_projects,
    update_project_status,
)
from luwu.utils import file_util


def run_project(project):
    params = project["params"]
    model_save_path = params["model_save_path"]
    log_path = os.path.join(
        model_save_path,
        f"luwu-classification-project-{project['id']}",
        f"train.log",
    )
    file_util.mkdirs(os.path.dirname(log_path))
    curdir = os.path.abspath(os.path.dirname(__file__))
    cd_path = os.path.abspath(os.path.join(os.path.join(curdir, ".."), ".."))
    py_path = os.path.join("." + curdir[len(cd_path) :], "train_project.py")
    cmd = f"""cd {cd_path};python {py_path} {project['id']} > {log_path} 2>&1"""
    st, out = subprocess.getstatusoutput(cmd)
    if st == 0:
        logger.info("处理成功！")
    else:
        logger.info("处理失败！")
        raise Exception(out)


def run():
    # 项目编号->进程句柄的映射
    xid_process_map = {}
    while True:
        projects = read_train_projects()
        cnt = 0
        for xid, p in xid_process_map.items():
            if p.is_alive():
                logger.debug(f"项目 {xid}, 进程 {p.pid}, 活跃中...")
                cnt += 1
        logger.info(f"当前活跃进程数 {cnt}!")
        # logger.info(projects)
        # 按照返回的信息，纠正每个任务的状态
        for project in projects:
            xid = project["id"]
            status = project["status"]
            # 检查不该处于运行状态的任务，是否有运行的。
            # 如果有，就终止它
            if status not in (1, 2):
                # logger.info(f"!1 {project}")
                p: multiprocessing.Process = xid_process_map.get(xid, None)
                if p and p.is_alive():
                    logger.info(
                        f"发现一个异常运行的任务 project {xid} | process {p.pid}，尝试 terminate..."
                    )
                    p.terminate()
                if p and p.is_alive():
                    logger.info(
                        f"发现一个异常运行的任务 project {xid} | process {p.pid}，尝试 kill..."
                    )
                    p.kill()
                if xid in xid_process_map:
                    del xid_process_map[xid]
            # 检查是否存在待处理任务
            # 如果有，并且没有处于运行状态，就运行它
            if status == 1:
                logger.info(f"发现一个待处理任务 project {xid}...")
                p: multiprocessing.Process = xid_process_map.get(xid, None)
                if p:
                    if p.is_alive():
                        logger.info(f"任务已经存在 project {xid} | process {p.pid}！跳过")
                    # 进程正常退出
                    elif p.exitcode == 0:
                        update_project_status(xid, 3)
                        del xid_process_map[xid]
                    # 进程异常退出
                    else:
                        update_project_status(xid, 6)
                        del xid_process_map[xid]
                else:
                    logger.info(f"创建一个新的任务 project {xid}...")
                    p = multiprocessing.Process(target=run_project, args=(project,))
                    p.start()
                    xid_process_map[xid] = p
                    logger.info(f"创建完成！project {xid} | process {p.pid}")
            # 检查处于运行中状态的任务
            # 如果任务已经结束了，就更新状态
            if status == 2:
                p: multiprocessing.Process = xid_process_map.get(xid, None)
                if p:
                    # 进程存活
                    if p.is_alive():
                        continue
                    # 进程正常退出
                    elif p.exitcode == 0:
                        update_project_status(xid, 3)
                        del xid_process_map[xid]
                    # 进程异常退出
                    else:
                        update_project_status(xid, 6)
                        del xid_process_map[xid]
                else:
                    update_project_status(xid, 6)
        # 对于已经停止的、且处理逻辑删除状态的任务，执行物理删除
        for project in projects:
            xid = project["id"]
            deleted = project["deleted"]
            if deleted and xid not in xid_process_map:
                delete_project_by_id(xid)
        time.sleep(10)


if __name__ == "__main__":
    run()

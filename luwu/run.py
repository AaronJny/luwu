# -*- coding: utf-8 -*-
# @Author       : AaronJny
# @LastEditTime : 2021-01-29
# @FilePath     : /LuWu/luwu/run.py
# @Desc         :
import time
import traceback
from multiprocessing import Process
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tensorflow as tf
from loguru import logger

# 引入顺序不能变动，必须先执行此段代码，才能引入luwu下的模块
gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from luwu.backend import app
from luwu.scripts import scheduler
from luwu.backend.config import Config


def init_luwu_dir():
    dir_name = Config.LUWU_DIR
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def run():
    while True:
        processes = []
        init_luwu_dir()
        try:
            # 启动Web服务
            logger.info("正在启动 Web 服务进程...")
            web_process = Process(target=app.run)
            processes.append(web_process)
            web_process.start()

            time.sleep(20)

            # 启动调度进程
            logger.info("正在启动任务调度进程...")
            scheduler_process = Process(target=scheduler.run)
            processes.append(scheduler_process)
            scheduler_process.start()

            for process in processes:
                process.join()

        except KeyboardInterrupt:
            logger.info("收到终止信号，正在关闭所有进程...")
            for process in processes:
                if process.is_alive():
                    process.terminate()
                if process.is_alive():
                    process.kill()
            logger.info("关闭完成！结束程序！")
            break
        except Exception as e:
            logger.error(e)
            logger.error(traceback.format_exc())
            # 收到终止信号或抛出异常时，关闭所有进程后再退出
            logger.info("出现异常！正在关闭所有进程...")
            for process in processes:
                if process.is_alive():
                    process.terminate()
                if process.is_alive():
                    process.kill()
            logger.info("关闭完成！正在重试...！")
            time.sleep(10)


if __name__ == "__main__":
    run()

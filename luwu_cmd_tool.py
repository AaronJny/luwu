# -*- coding: utf-8 -*-
# @Author       : AaronJny
# @LastEditTime : 2021-03-09
# @FilePath     : /LuWu/luwu_cmd_tool.py
# @Desc         :
import argparse

parser = argparse.ArgumentParser(
    description="""
    Luwu命令行工具。
    你完全可以不写任何代码，就能够完成深度学习任务的开发。
    
    LuWu command tool.
    You can complete the development of deep learning tasks without writing any code.
    """
)
usage = """
usage: luwu_cmd_tool.py cmd

Luwu命令行工具。
你完全可以不写任何代码，就能够完成深度学习任务的开发。

LuWu command tool.
You can complete the development of deep learning tasks
without writing any code.

arguments:
    cmd: 命令参数，用以指定 Luwu 将要执行的操作
"""
# parser.add_argument(
#     "run web-server",
#     help="Run LuWu in the form of a web server.The default address is http://localhost:7788/ .",
#     type=str,
#     default="",
# )
subparsers = parser.add_subparsers(title="cmd", description="命令参数，用以指定 Luwu 将要执行的操作")

parse_web_server = subparsers.add_parser(
    "web-server", help="以web服务的形式启动Luwu，从而通过图形界面完成深度学习任务的开发。"
)
parse_web_server.set_defaults(cmd="web-server")

parse_object_detection = subparsers.add_parser("detection", help="通过命令行进行目标检测任务训练。")
parse_object_detection.set_defaults(cmd="detection")
parse_object_detection.add_argument(
    "--origin_dataset_path", help="处理前的数据集路径", type=str, default=""
)
parse_object_detection.add_argument(
    "--tfrecord_dataset_path", help="处理后的tfrecord数据集路径", type=str, default=""
)
parse_object_detection.add_argument(
    "--label_map_path", help="目标检测类别映射表(pbtxt)", type=str, default=""
)
parse_object_detection.add_argument(
    "--do_fine_tune", help="是否在预训练模型的基础上进行微调", type=bool, default=True
)
parse_object_detection.add_argument(
    "--fine_tune_checkpoint_path", help="预训练权重路径", type=str, default=""
)
parse_object_detection.add_argument(
    "--fine_tune_model_name",
    help='预训练模型名称。可选值的列表为["SSD ResNet50 V1 FPN 640x640 (RetinaNet50)","EfficientDet D0 512x512","CenterNet Resnet101 V1 FPN 512x512","CenterNet HourGlass104 512x512"]',
    type=str,
    default="",
)
parse_object_detection.add_argument(
    "--model_save_path", help="模型保存路径", type=str, default=""
)
parse_object_detection.add_argument(
    "--batch_size", help="mini batch 大小。默认 8.", type=int, default=8
)
parse_object_detection.add_argument(
    "--steps", help="训练steps数量. Defaults to 2000.", type=int, default=2000
)
parse_object_detection.add_argument(
    "--project_id", help="项目编号. Defaults to 0.", type=int, default=0
)

# TODO:增加分类任务的命令行执行
args = parser.parse_args()
print(args)
print(args._get_args())
print(args._get_kwargs())

if args.cmd == "web-server":
    from luwu import run

    run.run()
elif args.cmd == "detection":
    from luwu.core.models.image import LuWuTFModelsObjectDetector

    LuWuTFModelsObjectDetector(**dict(args._get_kwargs()))
else:
    print("请检查指令是否有误！")

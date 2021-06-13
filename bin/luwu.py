# -*- coding: utf-8 -*-
# @Author       : AaronJny
# @LastEditTime : 2021-06-13
# @FilePath     : /LuWu/bin/luwu.py
# @Desc         :
import os

os.environ["TF_KERAS"] = "1"
import argparse
import ast

parser = argparse.ArgumentParser(
    description="""
    Luwu命令行工具。
    你完全可以不写任何代码，就能够完成深度学习任务的开发。
    
    LuWu command tool.
    You can complete the development of deep learning tasks without writing any code.
    """
)
usage = """
usage: luwu --help

Luwu命令行工具。
你完全可以不写任何代码，就能够完成深度学习任务的开发。

LuWu command tool.
You can complete the development of deep learning tasks
without writing any code.

arguments:
    cmd: 命令参数，用以指定 Luwu 将要执行的操作
"""
parser.add_argument(
    "--version", "-v", action="version", version="Luwu 0.32", help="查看当前luwu版本号"
)
parser.add_argument(
    "--luwu_version", help="kaggle上使用的Luwu版本，不指定则默认使用最新的正式版本", type=str, default=""
)
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
    "--do_fine_tune", help="是否在预训练模型的基础上进行微调", type=ast.literal_eval, default=True
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
parse_object_detection.add_argument(
    "--run_with_kaggle",
    help="是否使用kaggle环境运行。必须先安装并配置kaggle api,才可以使用此选项。默认为False，即本地运行",
    type=ast.literal_eval,
    default=False,
)
parse_object_detection.add_argument(
    "--kaggle_accelerator",
    help="是否使用kaggle GPU进行加速（注意，仅当 run_with_kaggle 为 True 时此选项才有效）。默认不使用（即使用CPU）",
    type=ast.literal_eval,
    default=False,
)

parse_classification = subparsers.add_parser("classification", help="通过命令行进行图像分类任务训练。")
parse_classification.set_defaults(cmd="classification")
parse_classification.add_argument(
    "network_name",
    help="分类器名称，支持的分类器有：[LuwuDenseNet121ImageClassifier,\
        LuwuDenseNet169ImageClassifier,LuwuDenseNet201ImageClassifier,\
        LuwuVGG16ImageClassifier,LuwuVGG19ImageClassifier,LuwuMobileNetImageClassifier,\
        LuwuMobileNetV2ImageClassifier,LuwuInceptionResNetV2ImageClassifier,\
        LuwuInceptionV3ImageClassifier,LuwuNASNetMobileImageClassifier,\
        LuwuNASNetLargeImageClassifier,LuwuResNet50ImageClassifier,\
        LuwuResNet50V2ImageClassifier,LuwuResNet101ImageClassifier,\
        LuwuResNet101V2ImageClassifier,LuwuResNet152ImageClassifier,\
        LuwuResNet152V2ImageClassifier,LuwuMobileNetV3SmallImageClassifier,\
        LuwuMobileNetV3LargeImageClassifier,LuwuXceptionImageClassifier,\
        LuwuEfficientNetB0ImageClassifier,LuwuEfficientNetB1ImageClassifier,\
        LuwuEfficientNetB2ImageClassifier,LuwuEfficientNetB3ImageClassifier,\
        LuwuEfficientNetB4ImageClassifier,LuwuEfficientNetB5ImageClassifier,\
        LuwuEfficientNetB6ImageClassifier,LuwuEfficientNetB7ImageClassifier]",
)
parse_classification.add_argument(
    "--origin_dataset_path", help="处理前的数据集路径", type=str, default=""
)
parse_classification.add_argument(
    "--validation_dataset_path",
    help="验证数据集路径。如不指定，则从origin_dataset_path中进行切分。",
    type=str,
    default="",
)
parse_classification.add_argument(
    "--test_dataset_path",
    help="测试数据集路径。如不指定，则从origin_dataset_path中进行切分。",
    type=str,
    default="",
)
parse_classification.add_argument(
    "--tfrecord_dataset_path", help="处理后的tfrecord数据集路径", type=str, default=""
)
parse_classification.add_argument(
    "--model_save_path", help="模型保存路径", type=str, default=""
)
parse_classification.add_argument(
    "--validation_split", help="验证集切割比例。默认 0.1", type=float, default=0.1
)
parse_classification.add_argument(
    "--test_split", help="测试集切割比例。默认 0.1", type=float, default=0.1
)
parse_classification.add_argument(
    "--do_fine_tune",
    help="是进行fine tune，还是重新训练。默认 False",
    type=ast.literal_eval,
    default=False,
)
parse_classification.add_argument(
    "--freeze_epochs_ratio",
    help="当进行fine_tune时，会先冻结预训练模型进行训练一定epochs，再解冻全部参数训练一定epochs，此参数表示冻结训练epochs占全部epochs的比例（此参数仅当 do_fine_tune = True 时有效）。默认 0.1（当总epochs>1时，只要设置了比例，至少会训练一个epoch）",
    type=float,
    default=0.1,
)
parse_classification.add_argument(
    "--batch_size", help="mini batch 大小。默认 32.", type=int, default=8
)
parse_classification.add_argument(
    "--epochs", help="训练epoch数。默认 30.", type=int, default=30
)
parse_classification.add_argument(
    "--learning_rate", "-lr", help="学习率。默认 0.001.", type=float, default=0.001
)
parse_classification.add_argument(
    "--optimizer",
    help="训练时的优化器类型,可选参数为 [Adam, Adamax, Adagrad, Nadam, Adadelta, SGD, RMSprop]。默使用 Adam.",
    type=str,
    default="Adam",
)
parse_classification.add_argument(
    "--project_id", help="项目编号. Defaults to 0.", type=int, default=0
)
parse_classification.add_argument(
    "--image_augmentation_random_flip_horizontal",
    help="数据增强选项，是否做随机左右镜像。默认False.",
    type=ast.literal_eval,
    default=False,
)
parse_classification.add_argument(
    "--image_augmentation_random_flip_vertival",
    help="数据增强选项，是否做随机上下镜像。默认False.",
    type=ast.literal_eval,
    default=False,
)
parse_classification.add_argument(
    "--image_augmentation_random_crop",
    help="数据增强选项，是否做随机剪裁，剪裁尺寸为原来比例的0.9。默认False.",
    type=ast.literal_eval,
    default=False,
)
parse_classification.add_argument(
    "--image_augmentation_random_brightness",
    help="数据增强选项，是否做随机饱和度调节。默认False.",
    type=ast.literal_eval,
    default=False,
)
parse_classification.add_argument(
    "--image_augmentation_random_hue",
    help="数据增强选项，是否做随机色调调节。默认False.",
    type=ast.literal_eval,
    default=False,
)
parse_classification.add_argument(
    "--run_with_kaggle",
    help="是否使用kaggle环境运行。必须先安装并配置kaggle api,才可以使用此选项。默认为False，即本地运行",
    type=ast.literal_eval,
    default=False,
)
parse_classification.add_argument(
    "--kaggle_accelerator",
    help="是否使用kaggle GPU进行加速（注意，仅当 run_with_kaggle 为 True 时此选项才有效）。默认不使用（即使用CPU）",
    type=ast.literal_eval,
    default=False,
)

parse_text_classification = subparsers.add_parser(
    "text_classification", help="通过命令行进行文本分类任务训练。"
)
parse_text_classification.set_defaults(cmd="text_classification")
parse_text_classification.add_argument(
    "--origin_dataset_path", help="处理前的数据集路径", type=str, default=""
)
parse_text_classification.add_argument(
    "--validation_dataset_path",
    help="验证数据集路径。如不指定，则从origin_dataset_path中进行切分。",
    type=str,
    default="",
)
parse_text_classification.add_argument(
    "--test_dataset_path",
    help="测试数据集路径。如不指定，则从origin_dataset_path中进行切分。",
    type=str,
    default="",
)
parse_text_classification.add_argument(
    "--model_save_path", help="模型保存路径", type=str, default=""
)
parse_text_classification.add_argument(
    "--validation_split", help="验证集切割比例。默认 0.1", type=float, default=0.1
)
parse_text_classification.add_argument(
    "--test_split", help="测试集切割比例。默认 0.1", type=float, default=0.1
)
parse_text_classification.add_argument(
    "--frezee_pre_trained_model",
    help="在训练下游网络时，是否冻结预训练模型权重。默认 False",
    type=ast.literal_eval,
    default=False,
)
parse_text_classification.add_argument(
    "--batch_size", help="mini batch 大小。默认 32.", type=int, default=8
)
parse_text_classification.add_argument(
    "--epochs", help="训练epoch数。默认 10.", type=int, default=10
)
parse_text_classification.add_argument(
    "--learning_rate", "-lr", help="学习率。默认 0.001.", type=float, default=0.001
)
parse_text_classification.add_argument(
    "--optimizer",
    help="训练时的优化器类型,可选参数为 [Adam, Adamax, Adagrad, Nadam, Adadelta, SGD, RMSprop]。默使用 Adam.",
    type=str,
    default="Adam",
)
parse_text_classification.add_argument(
    "--optimize_with_piecewise_linear_lr",
    help="是否使用分段的线性学习率进行优化。默认 False",
    type=ast.literal_eval,
    default=False,
)
parse_text_classification.add_argument(
    "--project_id", help="项目编号. Defaults to 0.", type=int, default=0
)
parse_text_classification.add_argument(
    "--maxlen", help="单行文本允许的最大长度。默认 128.", type=int, default=128
)
parse_text_classification.add_argument(
    "--do_sample_balance",
    help='是否对数据集做样本均衡，允许传递三个值，""表示不进行样本均衡，"over"表示上采样（过采样），"under"表示下采样（欠采样）。默认 ""',
    type=str,
    default="",
)
parse_text_classification.add_argument(
    "--simplified_tokenizer",
    help="是否对分词器的词表进行精简。默认 False",
    type=ast.literal_eval,
    default=False,
)
parse_text_classification.add_argument(
    "--pre_trained_model_type",
    help='使用何种预训练模型。默认 "bert_base"',
    type=str,
    default="bert_base",
)
parse_text_classification.add_argument(
    "--language",
    help='预训练语料的语言。默认 "bert_base"',
    type=str,
    default="chinese",
)
parse_text_classification.add_argument(
    "--run_with_kaggle",
    help="是否使用kaggle环境运行。必须先安装并配置kaggle api,才可以使用此选项。默认为False，即本地运行",
    type=ast.literal_eval,
    default=False,
)
parse_text_classification.add_argument(
    "--kaggle_accelerator",
    help="是否使用kaggle GPU进行加速（注意，仅当 run_with_kaggle 为 True 时此选项才有效）。默认不使用（即使用CPU）",
    type=ast.literal_eval,
    default=False,
)


parse_text_sequence_labeling = subparsers.add_parser(
    "text_sequence_labeling", help="通过命令行进行文本序列标注任务训练。"
)
parse_text_sequence_labeling.set_defaults(cmd="text_sequence_labeling")
parse_text_sequence_labeling.add_argument(
    "--origin_dataset_path", help="处理前的数据集路径", type=str, default=""
)
parse_text_sequence_labeling.add_argument(
    "--validation_dataset_path",
    help="验证数据集路径。如不指定，则从origin_dataset_path中进行切分。",
    type=str,
    default="",
)
parse_text_sequence_labeling.add_argument(
    "--test_dataset_path",
    help="测试数据集路径。如不指定，则从origin_dataset_path中进行切分。",
    type=str,
    default="",
)
parse_text_sequence_labeling.add_argument(
    "--model_save_path", help="模型保存路径", type=str, default=""
)
parse_text_sequence_labeling.add_argument(
    "--validation_split", help="验证集切割比例。默认 0.1", type=float, default=0.1
)
parse_text_sequence_labeling.add_argument(
    "--test_split", help="测试集切割比例。默认 0.1", type=float, default=0.1
)
parse_text_sequence_labeling.add_argument(
    "--frezee_pre_trained_model",
    help="在训练下游网络时，是否冻结预训练模型权重。默认 False",
    type=ast.literal_eval,
    default=False,
)
parse_text_sequence_labeling.add_argument(
    "--batch_size", help="mini batch 大小。默认 32.", type=int, default=8
)
parse_text_sequence_labeling.add_argument(
    "--epochs", help="训练epoch数。默认 10.", type=int, default=10
)
parse_text_sequence_labeling.add_argument(
    "--learning_rate", "-lr", help="学习率。默认 2e-6.", type=float, default=2e-6
)
parse_text_sequence_labeling.add_argument(
    "--optimizer",
    help="训练时的优化器类型,可选参数为 [Adam, Adamax, Adagrad, Nadam, Adadelta, SGD, RMSprop]。默使用 Adam.",
    type=str,
    default="Adam",
)
parse_text_sequence_labeling.add_argument(
    "--optimize_with_piecewise_linear_lr",
    help="是否使用分段的线性学习率进行优化。默认 False",
    type=ast.literal_eval,
    default=False,
)
parse_text_sequence_labeling.add_argument(
    "--project_id", help="项目编号. Defaults to 0.", type=int, default=0
)
parse_text_sequence_labeling.add_argument(
    "--maxlen", help="单行文本允许的最大长度。默认 128.", type=int, default=128
)
parse_text_sequence_labeling.add_argument(
    "--simplified_tokenizer",
    help="是否对分词器的词表进行精简。默认 False",
    type=ast.literal_eval,
    default=False,
)
parse_text_sequence_labeling.add_argument(
    "--pre_trained_model_type",
    help='使用何种预训练模型。默认 "bert_base"',
    type=str,
    default="bert_base",
)
parse_text_sequence_labeling.add_argument(
    "--language",
    help='预训练语料的语言。默认 "bert_base"',
    type=str,
    default="chinese",
)
parse_text_sequence_labeling.add_argument(
    "--bert_layers",
    help="使用的bert模型层数。默认 12",
    type=int,
    default=12,
)
parse_text_sequence_labeling.add_argument(
    "--crf_lr_multiplier",
    help="CRF层的学习率缩放倍数。默认放大1000倍",
    type=float,
    default=1000.0,
)
parse_text_sequence_labeling.add_argument(
    "--run_with_kaggle",
    help="是否使用kaggle环境运行。必须先安装并配置kaggle api,才可以使用此选项。默认为False，即本地运行",
    type=ast.literal_eval,
    default=False,
)
parse_text_sequence_labeling.add_argument(
    "--kaggle_accelerator",
    help="是否使用kaggle GPU进行加速（注意，仅当 run_with_kaggle 为 True 时此选项才有效）。默认不使用（即使用CPU）",
    type=ast.literal_eval,
    default=False,
)


args = parser.parse_args()
print(args)


def run():
    if args.cmd == "web-server":
        from luwu import run

        run.run()
    elif args.cmd == "detection":
        from luwu.core.models.image import LuWuTFModelsObjectDetector

        if args.run_with_kaggle:
            from luwu.core.models.kaggle.kaggle import KaggleUtil

            KaggleUtil(LuWuTFModelsObjectDetector, **dict(args._get_kwargs())).run()
        else:
            LuWuTFModelsObjectDetector(**dict(args._get_kwargs())).run()
    elif args.cmd == "classification":
        class_name = args.network_name
        net_name = class_name.lstrip("Luwu").rstrip("ImageClassifier")
        import importlib

        luwu_image = importlib.import_module("luwu.core.models.image")
        classifier_class = getattr(luwu_image, class_name)
        params = dict(args._get_kwargs())
        params["net_name"] = net_name
        if args.run_with_kaggle:
            from luwu.core.models.kaggle.kaggle import KaggleUtil

            KaggleUtil(classifier_class, **params).run()
        else:
            classifier_class(**params).run()
    elif args.cmd == "text_classification":
        from luwu.core.models.text import TransformerTextClassification

        if args.run_with_kaggle:
            from luwu.core.models.kaggle.kaggle import KaggleUtil

            KaggleUtil(TransformerTextClassification, **dict(args._get_kwargs())).run()
        else:
            TransformerTextClassification(**dict(args._get_kwargs())).run()

    elif args.cmd == "text_sequence_labeling":
        from luwu.core.models.text import TransformerTextSequenceLabeling

        if args.run_with_kaggle:
            raise Exception("暂未支持在kaggle上运行此任务！")
        else:
            TransformerTextSequenceLabeling(**dict(args._get_kwargs())).run()
    else:
        print("请检查指令是否有误！")


if __name__ == "__main__":
    run()

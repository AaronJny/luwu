<!--
 * @Author       : AaronJny
 * @LastEditTime : 2021-03-17
 * @FilePath     : /LuWu/docs/二、[基本使用]了解 Luwu 的命令行模式.md
 * @Desc         : 
-->
# 二、了解 Luwu 的命令行模式

Luwu 最开始是作为一个 Web 开发的，但写着写着，我忽然发现命令行工具貌似更好用...

所以后续开发重点就先迁移到模型功能和命令行支持上面来了，Web界面就后面再说吧...

> 当前陆吾的命令行工具仅支持如下功能：
> - 启动 Web 服务
> - 执行目标检测任务
> - 执行图像分类任务

当我们安装完陆吾之后，陆吾的命令行工具会自动加入到系统的可执行环境中。打开终端，输入`luwu -h`并回车，即可查看陆吾的命令行工具的帮助信息：

```sh
luwu -h
```

output:

```
usage: luwu [-h] {web-server,detection,classification} ...

Luwu命令行工具。 你完全可以不写任何代码，就能够完成深度学习任务的开发。 LuWu command tool. You can complete the
development of deep learning tasks without writing any code.

optional arguments:
  -h, --help            show this help message and exit

cmd:
  命令参数，用以指定 Luwu 将要执行的操作

  {web-server,detection,classification}
    web-server          以web服务的形式启动Luwu，从而通过图形界面完成深度学习任务的开发。
    detection           通过命令行进行目标检测任务训练。
    classification      通过命令行进行图像分类任务训练。
```

可以看到，陆吾支持 `web-server`、`detection`和`classification`三种功能。

其中，`web-server`是用来启动web服务的，输入`luwu web-server`并回车即可。web服务如何使用这里先不说，后面补文档。

`detection`是目标检测相关功能，帮助信息如下：

```sh
luwu detection -h
```

output:

```
usage: luwu detection [-h] [--origin_dataset_path ORIGIN_DATASET_PATH]
                      [--tfrecord_dataset_path TFRECORD_DATASET_PATH]
                      [--label_map_path LABEL_MAP_PATH]
                      [--do_fine_tune DO_FINE_TUNE]
                      [--fine_tune_checkpoint_path FINE_TUNE_CHECKPOINT_PATH]
                      [--fine_tune_model_name FINE_TUNE_MODEL_NAME]
                      [--model_save_path MODEL_SAVE_PATH]
                      [--batch_size BATCH_SIZE] [--steps STEPS]
                      [--project_id PROJECT_ID]
                      [--run_with_kaggle RUN_WITH_KAGGLE]
                      [--kaggle_accelerator KAGGLE_ACCELERATOR]

optional arguments:
  -h, --help            show this help message and exit
  --origin_dataset_path ORIGIN_DATASET_PATH
                        处理前的数据集路径
  --tfrecord_dataset_path TFRECORD_DATASET_PATH
                        处理后的tfrecord数据集路径
  --label_map_path LABEL_MAP_PATH
                        目标检测类别映射表(pbtxt)
  --do_fine_tune DO_FINE_TUNE
                        是否在预训练模型的基础上进行微调
  --fine_tune_checkpoint_path FINE_TUNE_CHECKPOINT_PATH
                        预训练权重路径
  --fine_tune_model_name FINE_TUNE_MODEL_NAME
                        预训练模型名称。可选值的列表为["SSD ResNet50 V1 FPN 640x640
                        (RetinaNet50)","EfficientDet D0 512x512","CenterNet
                        Resnet101 V1 FPN 512x512","CenterNet HourGlass104
                        512x512"]
  --model_save_path MODEL_SAVE_PATH
                        模型保存路径
  --batch_size BATCH_SIZE
                        mini batch 大小。默认 8.
  --steps STEPS         训练steps数量. Defaults to 2000.
  --project_id PROJECT_ID
                        项目编号. Defaults to 0.
  --run_with_kaggle RUN_WITH_KAGGLE
                        是否使用kaggle环境运行。必须先安装并配置kaggle
                        api,才可以使用此选项。默认为False，即本地运行
  --kaggle_accelerator KAGGLE_ACCELERATOR
                        是否使用kaggle GPU进行加速（注意，仅当 run_with_kaggle 为 True
                        时此选项才有效）。默认不使用（即使用CPU）
```

如何使用？请参考文档 [五、[目标检测]训练一个目标检测模型](./五、[目标检测]训练一个目标检测模型.md)。

`classification`是目标检测相关功能，帮助信息如下：

```sh
luwu classification -h
```

output:

```
usage: luwu classification [-h] [--origin_dataset_path ORIGIN_DATASET_PATH]
                           [--tfrecord_dataset_path TFRECORD_DATASET_PATH]
                           [--model_save_path MODEL_SAVE_PATH]
                           [--validation_split VALIDATION_SPLIT]
                           [--do_fine_tune DO_FINE_TUNE]
                           [--batch_size BATCH_SIZE] [--epochs EPOCHS]
                           [--project_id PROJECT_ID]
                           [--run_with_kaggle RUN_WITH_KAGGLE]
                           [--kaggle_accelerator KAGGLE_ACCELERATOR]
                           network_name

positional arguments:
  network_name          分类器名称，支持的分类器有：[LuwuDenseNet121ImageClassifier, LuwuDen
                        seNet169ImageClassifier,LuwuDenseNet201ImageClassifier
                        , LuwuVGG16ImageClassifier,LuwuVGG19ImageClassifier,Lu
                        wuMobileNetImageClassifier, LuwuMobileNetV2ImageClassi
                        fier,LuwuInceptionResNetV2ImageClassifier, LuwuIncepti
                        onV3ImageClassifier,LuwuNASNetMobileImageClassifier, L
                        uwuNASNetLargeImageClassifier,LuwuResNet50ImageClassif
                        ier, LuwuResNet50V2ImageClassifier,LuwuResNet101ImageC
                        lassifier, LuwuResNet101V2ImageClassifier,LuwuResNet15
                        2ImageClassifier, LuwuResNet152V2ImageClassifier,LuwuM
                        obileNetV3SmallImageClassifier, LuwuMobileNetV3LargeIm
                        ageClassifier,LuwuXceptionImageClassifier, LuwuEfficie
                        ntNetB0ImageClassifier,LuwuEfficientNetB1ImageClassifi
                        er, LuwuEfficientNetB2ImageClassifier,LuwuEfficientNet
                        B3ImageClassifier, LuwuEfficientNetB4ImageClassifier,L
                        uwuEfficientNetB5ImageClassifier, LuwuEfficientNetB6Im
                        ageClassifier,LuwuEfficientNetB7ImageClassifier]

optional arguments:
  -h, --help            show this help message and exit
  --origin_dataset_path ORIGIN_DATASET_PATH
                        处理前的数据集路径
  --tfrecord_dataset_path TFRECORD_DATASET_PATH
                        处理后的tfrecord数据集路径
  --model_save_path MODEL_SAVE_PATH
                        模型保存路径
  --validation_split VALIDATION_SPLIT
                        验证集切割比例。默认 0.2
  --do_fine_tune DO_FINE_TUNE
                        是进行fine tune，还是重新训练。默认 False
  --batch_size BATCH_SIZE
                        mini batch 大小。默认 32.
  --epochs EPOCHS       训练epoch数。默认 30.
  --project_id PROJECT_ID
                        项目编号. Defaults to 0.
  --run_with_kaggle RUN_WITH_KAGGLE
                        是否使用kaggle环境运行。必须先安装并配置kaggle
                        api,才可以使用此选项。默认为False，即本地运行
  --kaggle_accelerator KAGGLE_ACCELERATOR
                        是否使用kaggle GPU进行加速（注意，仅当 run_with_kaggle 为 True
                        时此选项才有效）。默认不使用（即使用CPU）
```


如何使用？请参考文档 [三、[图像分类]训练一个图像分类器](./三、[图像分类]训练一个图像分类器.md)。

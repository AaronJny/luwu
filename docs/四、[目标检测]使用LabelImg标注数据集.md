<!--
 * @Author       : AaronJny
 * @LastEditTime : 2021-03-26
 * @FilePath     : /LuWu/docs/四、[目标检测]使用LabelImg标注数据集.md
 * @Desc         : 
-->
举个例子，看这么一张图片：

![](assets/1614761531915.jpg)

能看到图中有各种字母和图形，假设我们现在需要通过程序判断图片中的目标（字母或图形）的位置，以及目标是正对我们的、还是倾斜着侧对我们的。应该怎么做？

用目标检测是完全可以解决的。

我们只需要标注图像中每个目标的位置，以及目标是正对还是侧对，在目标检测模型上训练就可以了。

而LabelImg则是用的比较多的图像标注工具之一，让我们一起看一下怎么用吧！

首先，安装`LabelImg`。

macos和linux可以考虑使用`pip`进行安装：

```sh
pip install labelimg
```

windows的话，直接下载编译好的二进制文件可能更加方便。下载地址：

[https://github.com/tzutalin/labelImg/releases](https://github.com/tzutalin/labelImg/releases)

如果还不能解决的话，参考项目在`GitHub`上给出的官方安装说明：

[tzutalin/labelImg](https://github.com/tzutalin/labelImg)

安装完成后，打开
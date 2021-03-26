<!--
 * @Author       : AaronJny
 * @LastEditTime : 2021-03-26
 * @FilePath     : /LuWu/docs/一、[基本使用]安装Luwu.md
 * @Desc         : 
-->
# 一、[基本使用]安装 Luwu

Luwu 支持通过三种方式进行安装，分别为 使用 `pip` 安装、从源码进行安装、以及使用 `Docker`。需要注意的是，`Luwu` 暂不支持在 `Windows` 系统下运行。

下面分别进行介绍。

在此之前，先说一下 `Anaconda`。

**建议使用Anaconda管理Python环境**。

如果没有安装Anaconda，可以先自行安装。下载链接：[https://www.anaconda.com/products/individual#Downloads](https://www.anaconda.com/products/individual#Downloads)

当然，你不用 `Anaconda` 也可以，只是做一个推荐，只要能自己处理软件的依赖冲突问题即可。


## 1、使用 `pip` 进行安装 (`推荐`)

使用 Anaconda :

```sh
# 先创建虚拟环境
conda create -n luwu python=3.7
# 切换到虚拟环境
conda activate luwu
# 安装陆吾
pip install luwu -v
luwu -h
```

不使用 Anaconda :

```sh
pip3 install luwu -v
luwu -h
```

## 2、从源码进行安装

主要思路是从 `GitHub` 上拉取代码并安装。

使用 Anaconda：

```sh
conda create -n luwu python=3.7
conda activate luwu
git clone https://github.com/AaronJny/luwu.git
cd luwu
python setup.py install
luwu -h
```

不使用 Anaconda：

```sh
git clone https://github.com/AaronJny/luwu.git
cd luwu
python3 setup.py install
luwu -h
```

## 3、使用 `Docker` 环境运行

陆吾支持通过Docker运行，提供 `CPU` 和 `GPU` 两种运行环境。

首先，请需要安装好 `Docker`。没有安装的话，可以参考 [https://www.runoob.com/docker/ubuntu-docker-install.html](https://www.runoob.com/docker/ubuntu-docker-install.html)进行安装。

使用 `CPU` :

```sh
# 拉取最新镜像
docker pull aaronjny/luwu:latest
# 运行容器
docker run --name luwu -p 7788:7788 -v /home/xxx/data:/data aaronjny/luwu
```

使用 `GPU` :

`GPU`版本`Docker`镜像仅支持`Linux`系统。

如果要使用 `GPU` 版本的镜像，则需要：

1.在本机安装 Nvidia 显卡设备对应的最新版本的驱动

2.如果没有安装 `nvidia-container-toolkit`，需要先在本机安装 `nvidia-container-toolkit`。具体安装参考[https://nvidia.github.io/nvidia-docker/](https://nvidia.github.io/nvidia-docker/)。

以 `Debian` 系统的安装为例：

```sh
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
  sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install nvidia-container-toolkit
```

对于使用 `Deepin` 的朋友，可以使用如下的指令安装：

```sh
distribution="debian10"
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
  sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install nvidia-container-toolkit
```

3.拉取并运行`GPU`版本容器。

```sh
docker pull aaronjny/luwu:gpu
# 运行容器
docker run --name luwu --gpus all -p 7788:7788 -v /home/xxx/data:/data aaronjny/luwu:gpu
```

----------

## 总结

`pip` 安装和源码安装都是直接安装到本机，正常使用即可。`Docker`方式则比较特殊，后续会给出 luwu 在 `Docker` 环境下的优雅使用方式，在此之前可能就要自己多折腾了...
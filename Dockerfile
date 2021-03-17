# 基础镜像
FROM python:3.7-slim
# 创建文件夹
RUN mkdir /data
# 工作目录
WORKDIR /app
# 复制代码到镜像
COPY . /app
# 安装依赖
RUN sed -i 's#http://deb.debian.org#https://mirrors.aliyun.com#g' /etc/apt/sources.list
RUN apt-get update \
    && apt-get install -y --no-install-recommends git wget unzip build-essential\
    && apt-get purge -y --auto-remove \
    && rm -rf /var/lib/apt/lists/*
# Python依赖
RUN python setup.py install
# 设置时区
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 内部使用的端口
EXPOSE 7788

ENV PYTHONPATH /app

CMD ["python", "luwu/run.py"]
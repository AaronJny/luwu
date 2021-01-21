# 基础镜像
FROM python:3.7-slim
# 工作目录
WORKDIR /app
# 复制代码到镜像
COPY . /app
# 安装依赖
RUN pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
# 设置时区
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 内部使用的端口
EXPOSE 5000

ENV PYTHONPATH /app

CMD ["python", "test/test_kerastuner_image_classifier.py"]
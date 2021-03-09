# -*- coding: utf-8 -*-
# @Author       : AaronJny
# @LastEditTime : 2021-03-09
# @FilePath     : /LuWu/setup.py
# @Desc         :
from setuptools import setup, find_packages
from setuptools.command.install import install
from .install_script import run_install


class LuwuInstallCommand(install):
    def run(self):
        install.run(self)
        run_install()


setup(
    name="luwu",
    version="0.13",
    author="AaronJny",
    author_email="aaronjny7@gmail.com",
    description="LuWu——陆吾，一个简单的无代码深度学习平台。",
    url="https://github.com/AaronJny/luwu",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.6.5",
    cmdclass={
        "install": LuwuInstallCommand,
    },
)
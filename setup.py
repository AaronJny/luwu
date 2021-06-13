# -*- coding: utf-8 -*-
# @Author       : AaronJny
# @LastEditTime : 2021-06-13
# @FilePath     : /LuWu/setup.py
# @Desc         :
import os
import sys

import setuptools
from setuptools import find_packages, setup
from setuptools.command.install import install
from setuptools.command.install_scripts import install_scripts

from install_script import run_install


class LuwuInstallCommand(install):
    def run(self):
        install.run(self)
        run_install()


class LuwuInstallScripts(install_scripts):
    def run(self):
        setuptools.command.install_scripts.install_scripts.run(self)

        # Rename some script files
        for script in self.get_outputs():
            if script.endswith(".py") or script.endswith(".sh"):
                dest = script[:-3]
            else:
                continue
            print("moving %s to %s" % (script, dest))
            with open(script, "r") as f:
                content = f.read()
            header = f"#!{sys.executable}\n"
            self.write_script(os.path.split(dest)[-1], header + content)
            os.remove(script)


setup(
    name="luwu",
    version="0.32",
    author="AaronJny",
    author_email="aaronjny7@gmail.com",
    description="LuWu——陆吾，一个简单的无代码深度学习平台。",
    url="https://github.com/AaronJny/luwu",
    packages=find_packages(include=["luwu.*", "luwu", "bin/*"]),
    include_package_data=True,
    python_requires=">=3.6.5",
    scripts=["bin/luwu.py"],
    cmdclass={
        "install": LuwuInstallCommand,
        "install_scripts": LuwuInstallScripts,
    },
)

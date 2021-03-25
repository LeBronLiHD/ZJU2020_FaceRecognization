# -*- coding: UTF-8 -*-
import tarfile # 解压tgz文件用到的库
import os

def unzip():
    # 相对路径
    path = "./data/"
    # 打开tgz文件
    tar = tarfile.open(path + "lfw.tgz", "r:gz")

    for tarinfo in tar:
        # 解压缩
        tar.extract(tarinfo.name, r"./data/unzip")

    # 关闭tgz文件
    tar.close()
    print("unzip done!")

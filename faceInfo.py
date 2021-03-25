# -*- coding: UTF-8 -*-
import tarfile
import os

def docuFaceInfo():
    pathUnzip = "./data/unzip/lfw" # 解压数据的路径
    pathUnzipChoose = "./data/unzip/mine" # 人为选择的数据的路径

    # 遍历数据集，并存入inf列表
    # 建立txt文件记录数据
    face_info_txt = "./FaceInfo.txt"
    f = open(face_info_txt, 'w')

    # 遍历
    count = 0
    files = os.listdir(pathUnzipChoose)
    for filename in files:
        printFilename = "Name " + str(count) + " " + filename
        print(printFilename)
        count += 1

    index = 0

    # 写入txt文件中
    for filename in files:
        tempDir = pathUnzipChoose + "/" + filename
        tempFiles = os.listdir(tempDir)
        for tempFilename in tempFiles:
            fileDir = str(index) + " " + filename + " " + tempFilename + "\n"
            f.write(fileDir) # 编号 姓名 图片路径与图片名
        index += 1 # 编号+1

    print("face information documentation done!")
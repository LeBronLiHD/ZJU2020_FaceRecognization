# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mp # mpimg 用于读取图片
from PIL import Image
import numpy as np
import math
import os

from imageProcess import preprocess

img_h = 250 # 默认图片高度
img_w = 250 # 默认图片宽度
channel = 1 # 图片维度，由于经过灰度处理，故为一维

pathUnzipChoose = "./data/unzip/mine" # 所选的十个人的数据路径
faceInfoMatch = [] # 人名对
facePathMatch = [] # 路径对
idNum = [] # ID

def readImage():

    for i in range(10):
        idNum.append(str(i))

    files = os.listdir(pathUnzipChoose) # 遍历文件夹
    index = 0
    for fileName in files:
        faceInfoMatch.append([str(index),fileName]) # 姓名与数字相匹配
        tempDir = pathUnzipChoose + "/" + fileName
        tempfiles = os.listdir(tempDir) # 遍历
        for tempFileNames in tempfiles:
            tempFilePath = pathUnzipChoose + "/" + fileName + "/" + tempFileNames
            facePathMatch.append([idNum[index], tempFilePath]) # 路径与数字（即姓名）匹配
        index += 1

    imageIndex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # 记录每个数据集中的图片数量

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for x in facePathMatch:
        nameIndex = int(x[0])
        if imageIndex[nameIndex] < 37: # 1~36 为训练集数据
            x_train.append(x[1]) # 存放路径
            y_train.append(idNum[nameIndex]) # 存放路径对应照片的label（姓名）
            imageIndex[nameIndex] += 1 # 数量+1
        else:
            x_test.append(x[1]) # 37及以后视为测试及数据
            y_test.append(idNum[nameIndex]) # 存放路径对应照片的label（姓名）
    print("image data x_train, y_train, x_test, y_test done!")
    return x_train, y_train, x_test, y_test


# 根据路径读入所有的图片，同时根据全局规定的图片尺寸要求进行裁剪,并按照模型输入要求reshape
def returnImage(x_train, x_test):
    X_train = []
    for i in range(len(x_train)):
        printStringTrain = "x_train reading image data, current position: " + str(i)
        print(printStringTrain)
        X_train.append(preprocess(mp.imread(x_train[i]), img_h, img_w)) # 读取图片，preprocess为图像预处理
    X_train = np.array(X_train) # 数组化处理
    print("X_train.shape is： ", X_train.shape) # (370,250,250)
    X_train = X_train.reshape(X_train.shape[0], X_train[0].shape[0], X_train[0].shape[1], channel).astype('float32')
    # 与模型相匹配啦！
    # 需要reshape一下下！

    X_test = [] # 同理
    for i in range(len(x_test)):
        printStringTest = "x_test reading image data, current position: " + str(i)
        print(printStringTest)
        X_test.append(preprocess(mp.imread(x_test[i]),img_h, img_w))
    X_test = np.array(X_test)
    X_test = X_test.reshape(X_test.shape[0],X_test[0].shape[0],X_test[0].shape[1],channel).astype('float32')
    print("image read done!")
    return X_train, X_test

''' use for debug
def main():
    # unzip()
    # docuFaceInfo()
    x_train, y_train, x_test, y_test = readImage()
    x_train, x_test = returnImage(x_train, x_test)

# 调用函数
if __name__ == '__main__':
    main()
'''

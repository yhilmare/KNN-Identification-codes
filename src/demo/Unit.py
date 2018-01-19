'''
Created on 2018年1月14日

@author: IL MARE
'''
import kNN
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL import ImageFilter
import os

if __name__ == "__main__":
    kNN.handWritingTest()
#======================图像处理相关==================================
#     try:
#         img = Image.open(r"g:\52.jpg")
#         print(img.mode)
#         r, g, b = img.split()
#         img = Image.merge("RGB", (r, g, b))
#         region = img.crop((115, 15, 275, 175))
#         region= region.transpose(Image.ROTATE_270)
# #         img.paste(region, (115, 15, 275, 175))
#         print(img.size)
#         print(img.getpixel((4,4)))
#         fig = plt.figure("Test")
#         ax = fig.add_subplot(121)
#         ax.imshow(img)
#         img = img.filter(ImageFilter.BLUR)
#         img = img.point(lambda i : i * 1.5)
#         bx = fig.add_subplot(122)
#         bx.imshow(img)
#         plt.show()
#     except Exception as e:
#         print(e)
#===================对数据进行knn分类======================================
#     print(kNN.classify0.__annotations__)
#     kNN.datingClassTest()
#======================对约会数据图形化表示====================================
#     mpl.rcParams["xtick.labelsize"] = 6
#     mpl.rcParams["ytick.labelsize"] = 6
#     fig = plt.figure("Fig")
#     ax = fig.add_subplot(111)
#     ax.set_xlabel("x")
#     ax.set_ylabel("y")
#     ax.scatter(returnMat[:, 0], returnMat[:, 1], 15.0 * np.array(datingLabel), 15 * np.array(datingLabel))
#     ax.legend("x")
#     plt.show()
#======================最原始的knn算法==============================================
#     groups, labels = kNN.createDatSet()
#     print(kNN.classify0([1,1], groups, labels, 3))

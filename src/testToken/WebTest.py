'''
Created on 2018年1月18日

@author: IL MARE
'''
import TokenTest
from urllib.request import urlopen
from PIL import Image
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt 
from io import BytesIO
import requests
from bs4 import BeautifulSoup
import re

if __name__ == "__main__":
    try:
        dataSet, labels = TokenTest.readDataSet()
        stu_id = "*******"
        passwd = "*******"
        resp = requests.get("http://jiaowu.swjtu.edu.cn/servlet/GetRandomNumberToJPEG", timeout=5)
        tmp_bt = resp.content
        pattern = TokenTest.kNNidentify(dataSet, labels, BytesIO(tmp_bt))
        print(pattern)
        img = Image.open(BytesIO(tmp_bt))
        matrix = np.asarray(img)
        fig = plt.figure("TEST")
        plt.imshow(img)
        plt.show()
#         header = {"Connection": "keep-alive","Cache-Control": "max-age=0",
#                       "Origin": "http://jiaowu.swjtu.edu.cn","Upgrade-Insecure-Requests": "1", 
#                       "User-Agent": "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36",
#                       "Content-Type": "application/x-www-form-urlencoded",
#                       "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
#                       "Referer": "http://jiaowu.swjtu.edu.cn/service/login.jsp?user_type=student",
#                       "Accept-Encoding": "gzip, deflate","Accept-Language": "zh-CN,zh;q=0.8"}
#         param = {"url":"../servlet/UserLoginCheckInfoAction", "OperatingSystem":"", 
#                      "Browser":"", "user_id":stu_id, "password":passwd, "ranstring":pattern, 
#                      "user_type":"student", "btn1":""}
#         resp = requests.post("http://jiaowu.swjtu.edu.cn/servlet/UserLoginSQLAction", timeout=5, headers=header, params=param, cookies=resp.cookies)
#         bsObj = BeautifulSoup(resp.text, "html.parser")
#         elt = bsObj.find("body")
#         if re.search(r"登录成功", elt.text):
#             print("login success")
#         else:
#             print("login error:", "".join(re.findall(r"[\u4e00-\u9fa5]", elt.text)))
    except Exception as e:
        print(e)

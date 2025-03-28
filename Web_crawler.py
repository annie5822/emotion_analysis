# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 17:25:38 2024

@author: user
"""

# 導入 模組(module) 
import requests 
# 導入 BeautifulSoup 模組(module)：解析HTML 語法工具
import bs4
import openpyxl
from openpyxl import Workbook
import re


# 文章連結
URL = "https://ptt-info.tw/bbs/KoreaDrama/M.1727520399.A.961.html"
# 設定Header與Cookie
my_headers = {'cookie': 'over18=1;'}
# 發送get 請求 到 ptt 八卦版
response = requests.get(URL, headers = my_headers)


#  把網頁程式碼(HTML) 丟入 bs4模組分析
soup = bs4.BeautifulSoup(response.text,"html.parser")

## PTT 上方4個欄位
header = soup.find_all('span','f3 push-content')

# 作者
author = header[0].text
# 看版
board = header[1].text
# 標題
title = header[2].text
# 日期
date = header[3].text


## 查找所有html 元素 抓出內容
main_container = soup.find(id='main-container')
# 把所有文字都抓出來
all_text = main_container.text
# 把整個內容切割透過 "-- " 切割成2個陣列
pre_text = all_text.split('--')[1]

pre_text = re.sub(r'\d{2}/\d{2} \d{2}:\d{2}', '', pre_text)
# 把每段文字 根據 '\n' 切開
pre_text = re.sub(r'[→推][^:]+:', '', pre_text)


texts = pre_text.split('\n')
# 如果你爬多篇你會發現 
contents = texts[3:]
# 內容
     # 儲存檔案
file = Workbook()
worksheet = file.active
r = 1
for i in contents:
    worksheet.cell(row = r,column = 1,value =  i)
    r+=1
file.save("comment.xlsx")

content = '\n'.join(contents)



# 顯示
#print('作者：'+author)
#print('看板：'+board)
#print('標題：'+title)
#print('日期：'+date)
print(content)




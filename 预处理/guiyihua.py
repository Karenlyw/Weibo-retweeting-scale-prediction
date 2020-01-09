# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 13:26:50 2019

@author: Administrator
"""

from sklearn.preprocessing import MinMaxScaler
from matplotlib import gridspec
import numpy as np
import csv
import matplotlib.pyplot as plt
fans=[]
perday=[]
number=[]
category=[]
score=[]
period=[]
popular=[]
video=[]
#with open(r'C:\Users\Administrator\Desktop\text-cnn-master\text-cnn-master\data\svm_train(one-hot).csv','r',encoding='utf-8') as f:
with open(r'C:\Users\Administrator\Desktop\text-cnn-master\text-cnn-master\data\svm_data.csv','r',encoding='gbk') as f:
    lines=csv.reader(f)
    for line in lines:
        category.append(line[1])
        fans.append([line[2]])
        perday.append([line[3]])
       # number.append(line[0])
        popular.append(line[4])  
        video.append(line[5])
        period.append(line[6])
        score.append([line[7]])
       
        
           
mm = MinMaxScaler(-1,1)
mm_fans = mm.fit_transform(fans)
mm_perday = mm.fit_transform(perday)

mm_score=mm.fit_transform(score)

new_fans=[]
new_perday=[]
new_score=[]
#for f in mm_fans:
#    new_fans.append('%.3f'%f)
#for d in mm_perday:
#    new_perday.append('%.3f'%d)
for d in mm_score:
    new_score.append('%.8f'%d)
#print('data is ',new_fans)
#print('after Min Max ',new_perday)

with open(r"C:\Users\Administrator\Desktop\text-cnn-master\text-cnn-master\data\svm_train2.csv","w",encoding='utf-8',newline='')as file:
    writer=csv.writer(file)  
#    writer.writerow(['微博编号','传播类别','cnn得分','发布者粉丝数','发布者日均发博数','发布时间段','发布者热转率'])  
    writer.writerow(['传播类别','cnn得分','发布者粉丝数','发布者日均发博数','发','布','时','间','段','发布者热转率'])  
    for j in range(0,len(new_score)):  
        writer.writerow([category[j],new_score[j],"".join(fans[j]),"".join(perday[j]),period1[j],period2[j],period3[j],period4[j],period5[j],popular[j]])  
                  

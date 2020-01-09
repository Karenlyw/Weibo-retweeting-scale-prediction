# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 22:48:50 2019

@author: Administrator
"""

import csv
i=0
j=0
with open(r"C:\Users\Administrator\Desktop\time.csv","r",encoding='gbk')as file:
        csv_reader = csv.reader(file)
        rows=[]
        for row in csv_reader:
            time=row[0].split(" ")
            if(len(time)==2):
                clock=(time[1].split(":"))[0]
                print(clock)
                if(clock=="00"or clock=="01" or clock=="02" or clock=="03" or clock=="04" or clock=="05"):
                    row.append("凌晨")
                elif(clock=="06" or clock=="07" or clock=="08" or clock=="09" or clock=="10"):
                    row.append("早晨")
                elif(clock=="11"or clock=="12" or clock=="13"):
                    row.append("中午")
                elif(clock=="14"or clock=="15" or clock=="16" or clock=="17" or clock=="18" ):
                    row.append("下午")
                else:
                    row.append("晚上")
            else:
                continue

            rows.append(row)
         
        with open(r"C:\Users\Administrator\Desktop\time.csv","w",encoding='utf-8',newline='')as f:
            writer=csv.writer(f)  
            writer.writerows(rows)

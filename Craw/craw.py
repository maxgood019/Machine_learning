# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 22:15:55 2019

@author: user
"""
from datetime import date, timedelta
from urllib.request import urlopen
from dateutil import rrule
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import numpy as np
import json 
import time

def craw_one_month(stock_number,date):
    url = ("http://www.twse.com.tw/exchangeReport/STOCK_DAY?response=json&date="+
           ('date')+"&stockNo="+str(stock_number))
    data = json.loads(urlopen(url).read())
    return pd.DataFrame(data['data'],columns=data['fields'])




#iloc : iloc[:3] = slice the first three 
# loc : loc[:3]  = slice until and include label 3 
fig,ax1 = plt.subplots(1,1,figsize=(10,2))
output['收盤價'] = output['收盤價'].astype(float)
price = output.loc[:]['收盤價']
date =  output.loc[:]['日期']
plt.plot(date,price)
#output.loc[:]['收盤價'].plot(figsize=(18,8))
plt.xlabel('date')
plt.ylabel('price')

output = craw_one_month('2330','20190606')
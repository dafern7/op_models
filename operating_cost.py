# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 15:07:12 2019

@author: IST_1
"""
import pandas as pd
import numpy as np 


dayahead_2018_lbmp = pd.read_csv('OASIS_Day-Ahead_Market_Zonal_LBMP_NYC_2018.csv')
dayahead_2018_lbmp['total'] = (dayahead_2018_lbmp['DAM Zonal LBMP']+dayahead_2018_lbmp['DAM Zonal Losses']-dayahead_2018_lbmp['DAM Zonal Congestion'])
dayahead_2017_lbmp = pd.read_csv('OASIS_Day-Ahead_Market_Zonal_LBMP_NYC_2017.csv')
dayahead_2017_lbmp['total'] = (dayahead_2017_lbmp['DAM Zonal LBMP']+dayahead_2017_lbmp['DAM Zonal Losses']-dayahead_2017_lbmp['DAM Zonal Congestion'])
dayahead_2016_lbmp = pd.read_csv('OASIS_Day-Ahead_Market_Zonal_LBMP_NYC_2016.csv')
dayahead_2016_lbmp['total'] = (dayahead_2016_lbmp['DAM Zonal LBMP']+dayahead_2016_lbmp['DAM Zonal Losses']-dayahead_2016_lbmp['DAM Zonal Congestion'])
dayahead_2016_lbmp = dayahead_2016_lbmp.drop(dayahead_2016_lbmp.index[1416:1440])
dayahead_2015_lbmp = pd.read_csv('OASIS_Day-Ahead_Market_Zonal_LBMP_NYC_2015.csv')
dayahead_2015_lbmp['total'] = (dayahead_2015_lbmp['DAM Zonal LBMP']+dayahead_2015_lbmp['DAM Zonal Losses']-dayahead_2015_lbmp['DAM Zonal Congestion'])

total_dayahead_lbmp = pd.DataFrame(columns=['date','2015','2016','2017','2018','average'])
total_dayahead_lbmp = pd.DataFrame(pd.date_range(start='1/1/2018', end='1/1/2019',freq='H'))
total_dayahead_lbmp = total_dayahead_lbmp.drop(total_dayahead_lbmp.index[-1],axis=0)
total_dayahead_lbmp = total_dayahead_lbmp.rename(columns={0:'date'})
total_dayahead_lbmp['2015'] = dayahead_2015_lbmp.total
total_dayahead_lbmp['2016'] = dayahead_2016_lbmp.total
total_dayahead_lbmp['2017']= dayahead_2017_lbmp.total
total_dayahead_lbmp['2018'] = dayahead_2018_lbmp.total

total_dayahead_lbmp['average'] = total_dayahead_lbmp[['2015','2016','2017','2018']].sum(axis=1)/4
total_dayahead_lbmp['dayofyear'] = total_dayahead_lbmp.date.dt.dayofyear
charge_rate1 = (12.5*4/0.85)/10
charge_rate2 = (10*4/0.85)/10

cost1 = list()
cost2 = list()

for day in set(total_dayahead_lbmp.dayofyear):
    print(day)
    charging_time = total_dayahead_lbmp.loc[(total_dayahead_lbmp.dayofyear==day)&((total_dayahead_lbmp.date.dt.hour>=22)|(total_dayahead_lbmp.date.dt.hour<=7))].average.sum()
    cost1.append(charging_time*charge_rate1)
    cost2.append(charging_time*charge_rate2)
    
total_cost1 = sum(cost1)
total_cost2 = sum(cost2)

print("12.5 MW option 4 hour battery operation cost: ", total_cost1)
print("10 MW option 4 hour battery operation cost: ", total_cost2)
print("Charge Rate 12.5MW option: ", charge_rate1)
print("Charge Rate 10MW option: ", charge_rate2)

    
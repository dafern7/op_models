# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 12:36:20 2019

@author: IST_1
"""
import pandas as pd
import numpy as np
import random

ancillary_2017 = pd.read_csv('OASIS_Day-Ahead_Market_Ancillary_Services_2017.csv')
ancillary_2017 = ancillary_2017.loc[ancillary_2017['Zone Name']=='N.Y.C.']
ancillary_2017 = ancillary_2017.rename(columns={'Eastern Date Hour':'date'})
ancillary_2017.date = pd.to_datetime(ancillary_2017.date)
ancillary_2017 = ancillary_2017.reset_index(drop=True)


ancillary_2016 = pd.read_csv('OASIS_Day-Ahead_Market_Ancillary_Services_2016.csv')
ancillary_2016 = ancillary_2016.loc[ancillary_2016['Zone Name']=='N.Y.C.'].reset_index(drop=True)
ancillary_2016 = ancillary_2016.rename(columns={'Eastern Date Hour':'date'})
ancillary_2016.date = pd.to_datetime(ancillary_2016.date)
ancillary_2016 = ancillary_2016.reset_index(drop=True)
ancillary_2016 = ancillary_2016.drop(ancillary_2016.index[1416:1440])
ancillary_2016 = ancillary_2016.reset_index(drop=True)

ancillary_2015 = pd.read_csv('OASIS_Day-Ahead_Market_Ancillary_Services_2015.csv')
ancillary_2015 = ancillary_2015.loc[ancillary_2015['Zone Name']=='N.Y.C.']
ancillary_2015 = ancillary_2015.rename(columns={'Eastern Date Hour':'date'})
ancillary_2015.date = pd.to_datetime(ancillary_2015.date)
ancillary_2015 = ancillary_2015.reset_index(drop=True)

freq_reg = pd.DataFrame(pd.date_range(start='1/1/2018', end='1/1/2019',freq='H'))
freq_reg = freq_reg.drop(freq_reg.index[-1],axis=0)
freq_reg = freq_reg.rename(columns={0:'date'})
freq_reg['2015'] = ancillary_2015['DAM NYCA Regulation Capacity']
freq_reg['2016'] = ancillary_2016['DAM NYCA Regulation Capacity']
freq_reg['2017']= ancillary_2017['DAM NYCA Regulation Capacity']
freq_reg['average'] = freq_reg[['2015','2016','2017']].sum(axis=1)/3/1000

dayahead_2018_lbmp = pd.read_csv('OASIS_Day-Ahead_Market_Zonal_LBMP_NYC_2018.csv')
dayahead_2018_lbmp['total'] = (dayahead_2018_lbmp['DAM Zonal LBMP']+dayahead_2018_lbmp['DAM Zonal Losses']-dayahead_2018_lbmp['DAM Zonal Congestion'])/1000
dayahead_2017_lbmp = pd.read_csv('OASIS_Day-Ahead_Market_Zonal_LBMP_NYC_2017.csv')
dayahead_2017_lbmp['total'] = (dayahead_2017_lbmp['DAM Zonal LBMP']+dayahead_2017_lbmp['DAM Zonal Losses']-dayahead_2017_lbmp['DAM Zonal Congestion'])/1000
dayahead_2016_lbmp = pd.read_csv('OASIS_Day-Ahead_Market_Zonal_LBMP_NYC_2016.csv')
dayahead_2016_lbmp['total'] = (dayahead_2016_lbmp['DAM Zonal LBMP']+dayahead_2016_lbmp['DAM Zonal Losses']-dayahead_2016_lbmp['DAM Zonal Congestion'])/1000
dayahead_2016_lbmp = dayahead_2016_lbmp.drop(dayahead_2016_lbmp.index[1416:1440])
dayahead_2015_lbmp = pd.read_csv('OASIS_Day-Ahead_Market_Zonal_LBMP_NYC_2015.csv')
dayahead_2015_lbmp['total'] = (dayahead_2015_lbmp['DAM Zonal LBMP']+dayahead_2015_lbmp['DAM Zonal Losses']-dayahead_2015_lbmp['DAM Zonal Congestion'])/1000

total_dayahead_lbmp = pd.DataFrame(columns=['date','2015','2016','2017','2018','average'])
total_dayahead_lbmp = pd.DataFrame(pd.date_range(start='1/1/2018', end='1/1/2019',freq='H'))
total_dayahead_lbmp = total_dayahead_lbmp.drop(total_dayahead_lbmp.index[-1],axis=0)
total_dayahead_lbmp = total_dayahead_lbmp.rename(columns={0:'date'})
total_dayahead_lbmp['2015'] = dayahead_2015_lbmp.total
total_dayahead_lbmp['2016'] = dayahead_2016_lbmp.total
total_dayahead_lbmp['2017']= dayahead_2017_lbmp.total
total_dayahead_lbmp['2018'] = dayahead_2018_lbmp.total

total_dayahead_lbmp['average'] = total_dayahead_lbmp[['2015','2016','2017','2018']].sum(axis=1)/4
freq_reg_revenue = list()


charging_costs = (total_dayahead_lbmp.loc[(total_dayahead_lbmp.date.dt.day==1)&(total_dayahead_lbmp.date.dt.month==1)&(total_dayahead_lbmp.date.dt.hour<=5)].average.sort_values(ascending=True).reset_index(drop=True)).head(n=4).sum()

for month_num in set(freq_reg.date.dt.month):
    for j in set(freq_reg.loc[freq_reg.date.dt.month==month_num].date.dt.day):
        
        freq_reg_revenue.append(freq_reg.loc[(freq_reg.date.dt.month==month_num)&(freq_reg.date.dt.day==j)&(freq_reg.date.dt.hour>=0)].sort_values(by='average',ascending=False).reset_index(drop=True).head(n=22).average.sum())
        
money = sum(freq_reg_revenue) - charging_costs

print('Annual Frequency Regulation Money :', money)
        

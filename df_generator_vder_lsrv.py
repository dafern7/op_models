# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 11:10:21 2019

@author: Richard_Fu
"""

import helper
import pandas as pd

import datetime
import numpy as np
import random


lbmp = pd.read_csv('OASIS_Day-Ahead_Market_Zonal_LBMP_HUDVAL.csv')

#data1 = pd.read_csv('1.44504 MW_PVWatts_hourly.csv')
#data2 = pd.read_csv('400.14kW_PVWatts_hourly.csv') 
#data1 = data1.fillna(0)
#data2 = data2.fillna(0)
#solar1 = data1['AC System Output (W)']/1000
#solar2 = data2['AC System Output (W)']/1000
#solar = solar1+solar2


solar = helper.pv_data(address='1101 Kitchawan Road, Yorktown Heights, NY, 10598', solar_size_kw_dc = 1 , tilt=5)
icap_date_start = '2018-06-24 00:00:00'
icap_date_end = '2018-09-15 23:00:00'
icap_hour_start = 14
icap_hour_end = 18
icap_value = 0.31847
icap = (icap_date_start,icap_date_end,icap_hour_start,icap_hour_end,icap_value)

drv_date_start = '2019-06-24 00:00:00'
drv_date_end = '2019-09-15 23:00:00'
drv_hour_start = 14
drv_hour_end = 19
drv_value = 0.2218
drv = (drv_date_start,drv_date_end,drv_hour_start,drv_hour_end,drv_value)


def vder_df_generator(lbmp,solar,drv,env_cred=0.02741,com_cred=0,verbose=True):
    solar = solar.drop(columns={'month','day','hour'})
    solar_sum = solar.sum() 
    
    df = pd.DataFrame(pd.date_range(start='11/05/2018', end='11/05/2019',freq='H'))
    df = df.drop(df.index[-1],axis=0)
    df = df.rename(columns={0:'date'})

    
    df['day_of_week'] = df.date.dt.dayofweek
    df['solar'] = 0
    df['lbmp'] = (lbmp['DAM Zonal LBMP']+lbmp['DAM Zonal Losses']-lbmp['DAM Zonal Congestion'])/1000
    df['icap'] = 0.000
    df['drv'] = 0.000
    df['community_cred'] = pd.Series([com_cred for x in range(len(df.index))])
    df['env_cred'] = pd.Series([env_cred for x in range(len(df.index))])
    df['lsrv_event'] = 'No Event'
    df['lsrv_rate'] = 0.000
    
#    for i in df.index:
#        if (df.date[i]>=pd.to_datetime(icap[0]))&(df.date[i]<=pd.to_datetime(icap[1]))&(df.day_of_week[i]>=1)&(df.day_of_week[i]<=5)&(df.date.dt.hour[i]>=icap[2])&(df.date.dt.hour[i]<=icap[3]):
#            df.icap[i] = icap[4]
#        else:
#            pass
    start = list()       
    for i in df.index:
        if (df.date[i]>=pd.to_datetime(drv[0]))&(df.date[i]<=pd.to_datetime(drv[1]))&(df.day_of_week[i]>=1)&(df.day_of_week[i]<=5)&(df.date.dt.hour[i]>=drv[2])&(df.date.dt.hour[i]<drv[3]):
            df.drv[i] = drv[4]
            start.append(df.date[i])
            
        else:
            pass
  
    start = list(set(start))

    
    
    #choose 10 days to be
    #june 21 5-7pm
    #june 24 4-6pm
    #july 8 2-4pm
    #july 13 3-5pm
    #july 24 5-7pm
    #august 1 3-5pm
    #august 9 5-7pm
    #august 13 5-7pm
    #august 25 2-4pm
    #september 5 4-6pm
#    start = pd.to_datetime('2018-06-24 00:00:00').dayofyear
#    end = pd.to_datetime('2018-09-30 23:00:00').dayofyear
    end_hour=0.00
    days=list()
    
    for i in range(1,11):
        day = random.choice(start)
        start_hour = random.choice([14,15,16])
        if start_hour == 16:
            end_hour = 18
        else:        
            end_hour = start_hour+random.choice([1,2,3])
            
        df.loc[(df.date.dt.day==day.day)&(df.date.dt.month==day.month)&(df.date.dt.hour.isin(list(range(start_hour,end_hour+1)))), 'lsrv_rate'] = 3.96
        df.loc[(df.date.dt.day==day.day)&(df.date.dt.month==day.month)&(df.date.dt.hour.isin(list(range(start_hour,end_hour+1)))), 'lsrv_event'] = 'Event '+str(i) 
        days.append(day)
    if verbose:
        print(days)
        
        
    df['total'] = pd.Series([df.lbmp[x]+df.icap[x]+df.drv[x]+df.community_cred[x]+df.env_cred[x] for x in range(len(df.index))]) 
#    df.loc[df.lsrv_rate != 0, 'lsrv_rate'] = 0
    df = df.rename(columns={'solar':'solar_output_kw','total':'vder'}) 
    helper.save_data(df,'vder_test',overwrite=True)
    return df
    
df = vder_df_generator(lbmp,solar,drv)


# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 14:36:09 2019

@author: Richard_Fu
"""

import pandas as pd
import numpy as np
import random

realtime_2018_lbmp = pd.read_csv('OASIS_Real_Time_Dispatch_Zonal_LBMP_NYC_2018.csv')
realtime_2018_lbmp = realtime_2018_lbmp.rename(columns={'RTD End Time Stamp':'date'})
realtime_2018_lbmp.date = pd.to_datetime(realtime_2018_lbmp.date)
realtime_2018_lbmp = realtime_2018_lbmp[(realtime_2018_lbmp.date.dt.minute == 0)&(realtime_2018_lbmp.date.dt.second == 0)]
realtime_2018_lbmp = realtime_2018_lbmp.reset_index(drop=True)
realtime_2018_lbmp['total'] = realtime_2018_lbmp['RTD Zonal LBMP']+realtime_2018_lbmp['RTD Zonal Losses']-realtime_2018_lbmp['RTD Zonal Congestion']


realtime_2017_lbmp = pd.read_csv('OASIS_Real_Time_Dispatch_Zonal_LBMP_NYC_2017.csv')

realtime_2017_lbmp = realtime_2017_lbmp.rename(columns={'RTD End Time Stamp':'date'})
realtime_2017_lbmp.date = pd.to_datetime(realtime_2017_lbmp.date)
realtime_2017_lbmp = realtime_2017_lbmp[(realtime_2017_lbmp.date.dt.minute == 0)&(realtime_2017_lbmp.date.dt.second == 0)]
realtime_2017_lbmp = realtime_2017_lbmp.reset_index(drop=True)
realtime_2017_lbmp['total'] = realtime_2017_lbmp['RTD Zonal LBMP']+realtime_2017_lbmp['RTD Zonal Losses']-realtime_2017_lbmp['RTD Zonal Congestion']


realtime_2016_lbmp = pd.read_csv('OASIS_Real_Time_Dispatch_Zonal_LBMP_NYC_2016.csv')

realtime_2016_lbmp = realtime_2016_lbmp.rename(columns={'RTD End Time Stamp':'date'})
realtime_2016_lbmp.date = pd.to_datetime(realtime_2016_lbmp.date)
realtime_2016_lbmp = realtime_2016_lbmp[(realtime_2016_lbmp.date.dt.minute == 0)&(realtime_2016_lbmp.date.dt.second == 0)]
realtime_2016_lbmp = realtime_2016_lbmp.reset_index(drop=True)
realtime_2016_lbmp = realtime_2016_lbmp.drop(realtime_2016_lbmp.index[1415:1439])
realtime_2016_lbmp['total'] = realtime_2016_lbmp['RTD Zonal LBMP']+realtime_2016_lbmp['RTD Zonal Losses']-realtime_2016_lbmp['RTD Zonal Congestion']


realtime_2015_lbmp = pd.read_csv('OASIS_Real_Time_Dispatch_Zonal_LBMP_NYC_2015.csv')
realtime_2015_lbmp = realtime_2015_lbmp.rename(columns={'RTD End Time Stamp':'date'})
realtime_2015_lbmp.date = pd.to_datetime(realtime_2015_lbmp.date)
realtime_2015_lbmp = realtime_2015_lbmp[(realtime_2015_lbmp.date.dt.minute == 0)&(realtime_2015_lbmp.date.dt.second == 0)]
realtime_2015_lbmp = realtime_2015_lbmp.reset_index(drop=True)
realtime_2015_lbmp['total'] = realtime_2015_lbmp['RTD Zonal LBMP']+realtime_2015_lbmp['RTD Zonal Losses']-realtime_2015_lbmp['RTD Zonal Congestion']

total_realtime_lbmp = pd.DataFrame(columns=['date','2015','2016','2017','2018','average'])
total_realtime_lbmp = pd.DataFrame(pd.date_range(start='1/1/2018', end='1/1/2019',freq='H'))
total_realtime_lbmp = total_realtime_lbmp.drop(total_realtime_lbmp.index[-1],axis=0)
total_realtime_lbmp = total_realtime_lbmp.rename(columns={0:'date'})
total_realtime_lbmp['2015'] = realtime_2015_lbmp.total
total_realtime_lbmp['2016'] = realtime_2016_lbmp.total
total_realtime_lbmp['2017']= realtime_2017_lbmp.total
total_realtime_lbmp['2018'] = realtime_2018_lbmp.total

total_realtime_lbmp['average'] = total_realtime_lbmp[['2015','2016','2017','2018']].sum(axis=1)/4/1000



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

#ancillary_2018 = pd.read_csv('OASIS_Day-Ahead_Market_Ancillary_Services_2018.csv')
#ancillary_2018 = ancillary_2018.loc[ancillary_2018['Zone Name']=='N.Y.C.']
#ancillary_2018 = ancillary_2018.rename(columns={'Eastern Date Hour':'date'})
#ancillary_2018.date = pd.to_datetime(ancillary_2018.date)


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


sync = pd.DataFrame(pd.date_range(start='1/1/2018', end='1/1/2019',freq='H'))
sync = sync.drop(sync.index[-1],axis=0)
sync = sync.rename(columns={0:'date'})
sync['2015'] = ancillary_2015['DAM 10 Min Sync']
sync['2016'] = ancillary_2016['DAM 10 Min Sync']
sync['2017']= ancillary_2017['DAM 10 Min Sync']
sync['average'] = sync[['2015','2016','2017']].sum(axis=1)/3

a_sync = pd.DataFrame(pd.date_range(start='1/1/2018', end='1/1/2019',freq='H'))
a_sync = a_sync.drop(a_sync.index[-1],axis=0)
a_sync = a_sync.rename(columns={0:'date'})
a_sync['2015'] = ancillary_2015['DAM 10 Min Non Sync']
a_sync['2016'] = ancillary_2016['DAM 10 Min Non Sync']
a_sync['2017']= ancillary_2017['DAM 10 Min Non Sync']
a_sync['average'] = a_sync[['2015','2016','2017']].sum(axis=1)/3

thirty = pd.DataFrame(pd.date_range(start='1/1/2018', end='1/1/2019',freq='H'))
thirty = thirty.drop(thirty.index[-1],axis=0)
thirty = thirty.rename(columns={0:'date'})
thirty['2015'] = ancillary_2015['DAM 30 Min Non Sync']
thirty['2016'] = ancillary_2016['DAM 30 Min Non Sync']
thirty['2017']= ancillary_2017['DAM 30 Min Non Sync']
thirty['average'] = thirty[['2015','2016','2017']].sum(axis=1)/3

freq_reg = pd.DataFrame(pd.date_range(start='1/1/2018', end='1/1/2019',freq='H'))
freq_reg = freq_reg.drop(freq_reg.index[-1],axis=0)
freq_reg = freq_reg.rename(columns={0:'date'})
freq_reg['2015'] = ancillary_2015['DAM NYCA Regulation Capacity']
freq_reg['2016'] = ancillary_2016['DAM NYCA Regulation Capacity']
freq_reg['2017']= ancillary_2017['DAM NYCA Regulation Capacity']
freq_reg['average'] = freq_reg[['2015','2016','2017']].sum(axis=1)/3

averages = pd.DataFrame(pd.date_range(start='1/1/2018', end='1/1/2019',freq='H'))
averages = averages.drop(averages.index[-1],axis=0)
averages = averages.rename(columns={0:'date'})
averages['realtime_lbmp'] = total_realtime_lbmp.average
averages['dayahead_lbmp'] = total_dayahead_lbmp.average
averages['sync']= sync.average/1000
averages['a_sync'] = a_sync.average/1000
averages['thirty'] = thirty.average/1000
averages['freq_reg'] = freq_reg.average/1000
averages['month'] = averages.date.dt.month
averages['day'] = averages.date.dt.day
averages['hour'] = averages.date.dt.hour

total_sync_nyiso = list()
total_async_nyiso = list()
total_thirty_nyiso = list()
total_freq_reg_nyiso = list()
total_sync_no_nyiso = list()
total_async_no_nyiso = list()
total_thirty_no_nyiso = list()
total_freq_reg_no_nyiso = list()

for month_num in set(averages.month):
    january = averages.loc[averages.month==month_num]
    days = random.sample(range(1,january.day.max()+1),7)
    nyiso_call = pd.DataFrame(columns=['date','realtime_lbmp','dayahead_lbmp','sync','a_sync','thirty','freq_reg','month','day','hour'])
    no_nyiso_call = january
    for i in days:
        nyiso_call = pd.concat([nyiso_call,january.loc[january.day==i].reset_index(drop=True)])
        no_nyiso_call = no_nyiso_call[no_nyiso_call.day!=i]
    
    sync_revenue = list()
    sync_realtime_lbmp_revenue = list()
    async_revenue = list()
    async_realtime_lbmp_revenue = list()
    thirty_revenue = list()
    thirty_realtime_lbmp_revenue = list()
    sync_revenue2 = list()
    async_revenue2 = list()
    thirty_revenue2 = list()
    nyiso_day_cost = list()
    no_nyiso_day_cost = list()
    no_nyiso_day_revenue = list()
    regular_lbmp_revenue = list()
    regular_lbmp_cost = list()
    freq_reg_revenue = list()
    freq_reg_revenue2 = list()
    freq_reg_cost = list()
    
    discharge_hours = 4
    charging_hours = 5 #this equates to a 80% efficiency for a four hour battery
    contract_charge = 0 #$/kW for each month for distribution line >63kV
    
    for j in set(nyiso_call.day):
    
        sync_revenue.append(nyiso_call.loc[(nyiso_call.day==j)&(nyiso_call.hour>=10)&(nyiso_call.hour<=18)].sort_values(by='sync',ascending=False).reset_index(drop=True).head(n=discharge_hours).sync.sum())
        sync_realtime_lbmp_revenue.append(nyiso_call.loc[(nyiso_call.day==j)&(nyiso_call.hour>=10)&(nyiso_call.hour<=18)].sort_values(by='sync',ascending=False).reset_index(drop=True).head(n=discharge_hours).realtime_lbmp.sum())
        
        async_revenue.append(nyiso_call.loc[(nyiso_call.day==j)&(nyiso_call.hour>=10)&(nyiso_call.hour<=18)].sort_values(by='a_sync',ascending=False).reset_index(drop=True).head(n=discharge_hours).a_sync.sum())
        async_realtime_lbmp_revenue.append(nyiso_call.loc[(nyiso_call.day==j)&(nyiso_call.hour>=10)&(nyiso_call.hour<=18)].sort_values(by='a_sync',ascending=False).reset_index(drop=True).head(n=discharge_hours).realtime_lbmp.sum())
        
        thirty_revenue.append(nyiso_call.loc[(nyiso_call.day==j)&(nyiso_call.hour>=10)&(nyiso_call.hour<=18)].sort_values(by='thirty',ascending=False).reset_index(drop=True).head(n=discharge_hours).thirty.sum())
        thirty_realtime_lbmp_revenue.append(nyiso_call.loc[(nyiso_call.day==j)&(nyiso_call.hour>=10)&(nyiso_call.hour<=18)].sort_values(by='thirty',ascending=False).reset_index(drop=True).head(n=discharge_hours).realtime_lbmp.sum())
        
        freq_reg_revenue.append(nyiso_call.loc[(nyiso_call.day==j)&(nyiso_call.hour>=8)].sort_values(by='freq_reg',ascending=False).reset_index(drop=True).head(n=12).freq_reg.sum())
        freq_reg_cost.append((nyiso_call.loc[(nyiso_call.day==j)&(nyiso_call.hour<=7)].dayahead_lbmp.sort_values(ascending=True).reset_index(drop=True)).head(n=round(charging_hours/2)+1).sum())
        
        
        nyiso_day_cost.append((nyiso_call.loc[(nyiso_call.day==j)&(nyiso_call.hour<=7)].dayahead_lbmp.sort_values(ascending=True).reset_index(drop=True)).head(n=charging_hours).sum())
            
    total_sync_nyiso.append(sum(sync_revenue)-sum(nyiso_day_cost)+sum(sync_realtime_lbmp_revenue))
    total_async_nyiso.append(sum(async_revenue)-sum(nyiso_day_cost)+sum(async_realtime_lbmp_revenue))
    total_thirty_nyiso.append(sum(thirty_revenue)-sum(nyiso_day_cost) +sum(thirty_realtime_lbmp_revenue))
    total_freq_reg_nyiso.append(sum(freq_reg_revenue)-sum(freq_reg_cost))
    
    for k in set(no_nyiso_call.day):
        
        no_nyiso_day_revenue.append((no_nyiso_call.loc[(no_nyiso_call.day==k)&(no_nyiso_call.hour>=7)].realtime_lbmp.sort_values(ascending=False).reset_index(drop=True)).head(n=discharge_hours).sum())
        
        no_nyiso_day_cost.append((no_nyiso_call.loc[(no_nyiso_call.day==k)&(no_nyiso_call.hour<=7)].dayahead_lbmp.sort_values(ascending=True).reset_index(drop=True)).head(n=charging_hours).sum())
        
    
        sync_revenue2.append((no_nyiso_call.loc[(no_nyiso_call.day==k)&(no_nyiso_call.hour>=10)&(no_nyiso_call.hour<=18)].sync.sort_values(ascending=False).reset_index(drop=True)).head(n=discharge_hours).sum())
        async_revenue2.append((no_nyiso_call.loc[(no_nyiso_call.day==k)&(no_nyiso_call.hour>=10)&(no_nyiso_call.hour<=18)].a_sync.sort_values(ascending=False).reset_index(drop=True)).head(n=discharge_hours).sum())
        thirty_revenue2.append((no_nyiso_call.loc[(no_nyiso_call.day==k)&(no_nyiso_call.hour>=10)&(no_nyiso_call.hour<=18)].thirty.sort_values(ascending=False).reset_index(drop=True)).head(n=discharge_hours).sum())   
        freq_reg_revenue2.append((no_nyiso_call.loc[(no_nyiso_call.day==k)&(no_nyiso_call.hour>=8)].freq_reg.sort_values(ascending=False).reset_index(drop=True)).head(n=12).sum())  
        
        
    total_sync_no_nyiso.append(sum(sync_revenue2)-sum(no_nyiso_day_cost)+sum(no_nyiso_day_revenue))
    total_async_no_nyiso.append(sum(async_revenue2)-sum(no_nyiso_day_cost)+sum(no_nyiso_day_revenue))
    total_thirty_no_nyiso.append(sum(thirty_revenue2)-sum(no_nyiso_day_cost)+sum(no_nyiso_day_revenue)) 
    total_freq_reg_no_nyiso.append(sum(freq_reg_revenue2)-sum(no_nyiso_day_cost)+sum(no_nyiso_day_revenue))    
    


monthly_revenue_sync = list()
monthly_revenue_async = list()
monthly_revenue_thirty = list()
monthly_revenue_freq_reg = list()

for i in range(0, len(total_sync_no_nyiso)): 
    monthly_revenue_sync.append(total_sync_no_nyiso[i] + total_sync_nyiso[i]) 
    monthly_revenue_async.append(total_async_no_nyiso[i] + total_async_nyiso[i])
    monthly_revenue_thirty.append(total_thirty_no_nyiso[i] + total_thirty_nyiso[i])
    monthly_revenue_freq_reg.append(total_freq_reg_no_nyiso[i] + total_freq_reg_nyiso[i])
    
    
for l in set(january.day):
    
    regular_lbmp_revenue.append((january.loc[(january.day==l)&(january.hour>=7)].dayahead_lbmp.sort_values(ascending=False).reset_index(drop=True)).head(n=discharge_hours).sum())
    
    regular_lbmp_cost.append((january.loc[(january.day==l)&(january.hour<=7)].dayahead_lbmp.sort_values(ascending=True).reset_index(drop=True)).head(n=charging_hours).sum())

total_dayahead_lbmp_revenue = sum(regular_lbmp_revenue) - sum(regular_lbmp_cost)-contract_charge

print('Day Ahead LBMP only: ', total_dayahead_lbmp_revenue)
print('Ancillary Service (Synchronous Reserve): ', sum(monthly_revenue_sync)-12*contract_charge)
print('Ancillary Service (Asynchronous Reserve): ', sum(monthly_revenue_async)-12*contract_charge)
print('Ancillary Service (Thirty Day Reserve): ', sum(monthly_revenue_thirty)-12*contract_charge)
print('Ancillary Service (Frequency Regulation Capacity) (Is not exactly right): ', sum(monthly_revenue_freq_reg)-12*contract_charge)

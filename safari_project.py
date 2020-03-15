# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 17:06:41 2019

@author: IST_1
"""

import pandas as pd
from matplotlib import pyplot as plt
from op_models import peak_shave
import helper
import numpy as np
from calculator import Calculator
import sys, os
from contextlib import contextmanager

df = pd.read_excel('Safari Project Interval Data_August 2018-October 2019.xlsx')
df = df[0:35036]
df = df.rename(columns={'Time Beginning': 'date'})

df['month'] = df.date.dt.month
df['day'] = df.date.dt.day
df['hour'] = df.date.dt.hour
df['minute'] = df.date.dt.minute
df.sort_values(by=['month', 'day', 'hour','minute'], inplace=True)
df = df.reset_index(drop=True)
df.drop(columns=['month', 'day', 'hour','minute'], inplace=True)
    
power = 250
capacity = 500
df = df.rename(columns={'kW-Demand':'original_building_power_kw'})
df['original_building_energy_kwh'] = df.original_building_power_kw/4
solar = helper.pv_data(address='Yorktown Heights, NY', solar_size_kw_dc=1105.60, tilt=6)


index = pd.date_range(start='1/1/'+str(2018), periods=8760, freq='60T') 
solar.index=index
solar = solar.resample('15T').asfreq()   
solar = solar.interpolate()
solar = solar[0:-1]
solar = solar.reset_index(drop=True)
df['solar_output_kw'] = solar.solar_output_kw

#plt.figure(figsize=(15,5))
#plt.plot(df['kW-Demand'])
#plt.plot(df.solar_output_kw)


df["demand_rate"] = [0] * len(df.index)
df["demand_rate_category"] = [0] * len(df.index)

#df.solar_output_kw = 0
df['power_post_solar_kw'] = df.original_building_power_kw-df.solar_output_kw
df['energy_post_solar_kwh'] = (df.original_building_power_kw-df.solar_output_kw)/4


on_peak1 = ((df['date'].dt.month >= 6) & (df['date'].dt.month <= 9) & (df['power_post_solar_kw'] > 900))
on_peak2 = ((df['date'].dt.month >= 6) & (df['date'].dt.month <= 9) & (df['power_post_solar_kw'] <= 900))
off_peak1 = ((df['date'].dt.month <=5) | (df['date'].dt.month >= 10) & (df['power_post_solar_kw'] > 900))
off_peak2 = ((df['date'].dt.month <=5) | (df['date'].dt.month >= 10) & (df['power_post_solar_kw'] <= 900))

df.loc[on_peak1, "demand_rate"] = ((900*(17.61-15.91))/df['power_post_solar_kw'])+15.90 
df.loc[on_peak2, "demand_rate"] = 17.61
df.loc[off_peak1, "demand_rate"] = ((900*(14.07-12.35))/df['power_post_solar_kw'])+12.35 
df.loc[off_peak2, "demand_rate"] = 14.07
    
df.loc[df["demand_rate"] == ((900*(17.61-15.91))/df['power_post_solar_kw'])+15.90, "demand_rate_category"] = "on_peak"
df.loc[df["demand_rate"] == 17.61, "demand_rate_category"] = "on_peak"
df.loc[df["demand_rate"] == ((900*(14.07-12.35))/df['power_post_solar_kw'])+12.35, "demand_rate_category"] = "off_peak"
df.loc[df["demand_rate"] == 14.07, "demand_rate_category"] = "off_peak"


df['energy_rate'] = 0.0187


df2 = pd.DataFrame(columns=['intervals','solar','Ein','Eout','charge_state','system_out'])
test_dates_df = pd.DataFrame(columns=['date','original_building_power_kw','solar_output_kw','demand_rate_category','demand_rate',
                                          'power_post_solar_kw','output','energy_rate'])




original_bld_cost = dict.fromkeys(list(range(1, 12)))
only_solar_cost = dict.fromkeys(list(range(1, 12)))
solar_and_storage_cost = dict.fromkeys(list(range(1, 12)))
solar_only_savings = dict.fromkeys(list(range(1, 12)))
solar_and_storage_savings = dict.fromkeys(list(range(1, 12)))

peak_post_solarandstorage = list()
peak_orig_bld_power = list()
peak_post_solar = list()
peak_demand_rate = list()
peak_demand_rate2 = list()

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


for i in set(df.date.dt.month):         
    
    test_dates = df.loc[(df.date.dt.month == i)]#&(data.date.dt.dayofyear <= (i+1))]
    test_dates = test_dates.reset_index(drop=True)
    df_new,model = peak_shave(test_dates, power=power, capacity=capacity, eff=.8, itc=True, project_type='solar+storage', export=True)
    test_dates['power_post_solarandstorage_kw'] = df_new.system_out
    test_dates['Eout'] = df_new.Eout
    test_dates['energy_post_solarandstorage_kwh'] = df_new.system_out/4
    test_dates_df = pd.concat([test_dates_df,test_dates])
    
    peak_post_solarandstorage.append(test_dates.power_post_solarandstorage_kw.max())
    peak_post_solar.append(test_dates.power_post_solar_kw.max())
    peak_orig_bld_power.append(test_dates.original_building_power_kw.max())
    peak_demand_rate.append(test_dates.loc[(test_dates.power_post_solarandstorage_kw==test_dates.power_post_solarandstorage_kw.max())].demand_rate.max())
    peak_demand_rate2.append(test_dates.loc[test_dates.power_post_solar_kw==test_dates.power_post_solar_kw.max()].demand_rate.min())
    
    
    with suppress_stdout():
        month_bld_cost, month_storageandsolar, month_just_solar = helper.annual_savings(test_dates)

    original_bld_cost[i] = month_bld_cost
    only_solar_cost[i] = month_just_solar
    solar_and_storage_cost[i] = month_storageandsolar
    solar_only_savings[i] = month_bld_cost - month_just_solar
    solar_and_storage_savings[i] = month_bld_cost-month_storageandsolar

for key in original_bld_cost:
    if original_bld_cost[key] == None:
        original_bld_cost[key] = 0

for key in only_solar_cost:
    if only_solar_cost[key] == None:
        only_solar_cost[key] = 0

for key in solar_and_storage_cost:
    if solar_and_storage_cost[key] == None:
        solar_and_storage_cost[key] = 0

for key in solar_only_savings:
    if solar_only_savings[key] == None:
        solar_only_savings[key] = 0
        
for key in solar_and_storage_savings:
    if solar_and_storage_savings[key] == None:
        solar_and_storage_savings[key] = 0


helper.superimposed_bargraphs(original_bld_cost, only_solar_cost, solar_and_storage_cost, project_type='solar+storage')

helper.peak_day_plot(test_dates_df, start=0, finish=0, hourly_intervals=4, peak_month=2, peak_day=1, project_type='solar+storage') 

  
#demand_expenses = list()
#energy_expenses = list()
#total_expenses = list()
#
#for i in set(df.date.dt.month): 
#    test_dates = df.loc[(df.date.dt.month == i)]#&(data.date.dt.dayofyear <= (i+1))]
#    test_dates = test_dates.reset_index(drop=True)
#    df_new,model = peak_shave(test_dates, power=power, capacity=capacity, eff=.8, itc=True, project_type='solar+storage', export=True)
#    df2 = pd.concat([df2,df_new])
#    test_dates_df = pd.concat([test_dates_df,test_dates])
#    demand_expenses.append((model.demand_expenses()))
#    energy_expenses.append((model.energy_expenses()))
#    total_expenses.append(model.demand_expenses()+model.energy_expenses())

#    
test_dates_df = test_dates_df.reset_index(drop=True)    
df2_sample = test_dates_df.loc[(df.date.dt.month==2)]
df2_sample = df2_sample.reset_index(drop=True)

plt.figure(figsize=(15,5))
plt.plot(df2_sample.index,df2_sample.original_building_power_kw)
plt.plot(df2_sample.index,df2_sample.power_post_solar_kw)
plt.plot(df2_sample.index,df2_sample.power_post_solarandstorage_kw)
#nums = list(range(0,24))
#plt.xticks(np.arange(0, 24*4, step=4),nums)
plt.xlabel('Intervals')
plt.ylabel('Building Demand (kW)')
plt.title('Peak Shaving during Sample Month (August)')
plt.legend(['Without Storage System','Solar Only','With Solar and Storage System'])

print('Total Costs: ', sum(original_bld_cost.values()))
print('Solar Only Savings: ', sum(solar_only_savings.values()))
print('Storage Only Savings: ', sum(solar_and_storage_savings.values())-sum(solar_only_savings.values()))  
print('Solar and Storage Savings: ', sum(solar_and_storage_savings.values()))
     
total_demand_savings = sum([(a_i*d_i - b_i*c_i) for a_i, b_i, c_i, d_i in zip(peak_orig_bld_power, peak_post_solarandstorage, peak_demand_rate, peak_demand_rate2)])
total_kW_savings_storage = [(a_i - b_i) for a_i, b_i,in zip(peak_post_solar, peak_post_solarandstorage)]
total_kW_savings_solar = [(a_i - b_i) for a_i, b_i,in zip(peak_orig_bld_power, peak_post_solar)]


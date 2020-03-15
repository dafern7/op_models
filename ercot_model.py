# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 11:43:32 2019

@author: IST_1
"""

import pandas as pd
from op_models import vder_only
import numpy as np
from matplotlib import pyplot as plt

df = pd.read_excel('ERCOT_DALMP.xlsx')
zones = list(set(df.zone))

houston_lmp = df.loc[df.zone=='LZ_HOUSTON'].reset_index(drop=True)
west_lmp = df.loc[df.zone=='LZ_WEST'].reset_index(drop=True)
south_lmp = df.loc[df.zone=='LZ_SOUTH'].reset_index(drop=True)
aen_lmp = df.loc[df.zone=='LZ_AEN'].reset_index(drop=True)
lcra_lmp = df.loc[df.zone=='LZ_LCRA'].reset_index(drop=True)
cps_lmp = df.loc[df.zone=='LZ_CPS'].reset_index(drop=True)
north_lmp = df.loc[df.zone=='LZ_NORTH'].reset_index(drop=True)
raybn_lmp = df.loc[df.zone=='LZ_RAYBN'].reset_index(drop=True)

list_of_regions = [houston_lmp,south_lmp,aen_lmp,lcra_lmp,cps_lmp,north_lmp,raybn_lmp,west_lmp]


discharge_hours = 4
charging_hours = 5 #this equates to a 80% efficiency for a four hour battery

#for month in set(houston_lmp.Date.dt.month):
#    for day in set(houston_lmp.loc[houston_lmp.Date.dt.month==month].Date.dt.day):
#        revenue.append(houston_lmp.loc[(houston_lmp.Date.dt.day==day)&
#                                       (houston_lmp.Date.dt.hour>=7)&(nyiso_call.hour<=18)].sort_values(by='sync',ascending=False).reset_index(drop=True).head(n=discharge_hours).sync.sum())
regional_revenues = dict()
regional_income = dict()
regional_expenses = dict()
for region in list_of_regions:
    region = region.rename(columns={'price':'vder','Date':'date'})
    
    region.vder = region.vder/1000
    
    df_out,model = vder_only(region, power=10000, capacity=40000, max_discharges=1, eff=.9, project_type='storage only', itc=False, max_line_limit=20000)
    
    region['Ein'] = df_out.Ein
    region['Eout'] = df_out.Eout
    region['system_out'] = df_out.system_out
    region['income'] = region.Eout*region.vder
    region['expense'] = region.Ein*region.vder

    
    
    regional_revenues[region.zone[1]] =  int(-model.P())
    regional_income[region.zone[1]] = int(model.income())
    regional_expenses[region.zone[1]] = int(model.expenses())
    
#sorted_income = region.sort_values(by='income',ascending=False).reset_index(drop=True)
#sorted_expenses = region.sort_values(by='expense',ascending=False).reset_index(drop=True)
region['interval'] = list(range(1,8760))
#sorted_income.interval = (sorted_income.interval/(sorted_income.interval.max()))*100

#cumulative_income = np.cumsum(sorted_income.income)
#cumulative_expenses = np.cumsum(sorted_expenses.expense)

plt.figure(figsize=(15,5))
plt.plot(region.interval,region.income)
plt.xlabel('Percentage of the Year')
plt.ylabel('Cumulative Income')
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 14:44:21 2019

@author: IST_1
"""

import pandas as pd
import op_models
from matplotlib import pyplot as plt


intrepid_data = pd.read_excel('ISASM CustCare 2019 r2.xlsx')

intrepid_data['dayofyear'] = intrepid_data.Timestamp.dt.dayofyear

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
lbmp = total_dayahead_lbmp['average'].repeat(4).shift(-1)[0:24817].reset_index(drop=True)

intrepid_data['lbmp'] = lbmp/4/1000

intrepid_data = intrepid_data.drop(columns={'Period','Datehour'})
intrepid_data = intrepid_data.rename(columns={'Timestamp':'date','Con Ed kW':'original_building_power_kw'})
intrepid_data['energy_rate'] = 0.0079
#intrepid_data['energy_rate'] = intrepid_data.lbmp


intrepid_data["demand_rate1"] = [0] * len(intrepid_data.index)
intrepid_data["demand_rate_category1"] = [0] * len(intrepid_data.index)
intrepid_data["demand_rate2"] = [0] * len(intrepid_data.index)
intrepid_data["demand_rate_category2"] = [0] * len(intrepid_data.index)

demand_tariff_type = "SC9 Rate 2"
# conditions
intermediate_peak = ((intrepid_data['date'].dt.month <= 5) | (intrepid_data['date'].dt.month >= 10)) & (
        (intrepid_data['date'].dt.weekday >= 0) &
        (intrepid_data['date'].dt.weekday <= 5)) & \
          ((intrepid_data['date'].dt.hour >= 8) & (intrepid_data['date'].dt.hour < 22))
          
on_peak1 = ((intrepid_data['date'].dt.month >= 6) & (intrepid_data['date'].dt.month <= 9)) & (
        (intrepid_data['date'].dt.weekday >= 0) &
        (intrepid_data['date'].dt.weekday <= 5)) & \
          ((intrepid_data['date'].dt.hour >= 8) & (intrepid_data['date'].dt.hour < 18))
          
on_peak2 = ((intrepid_data['date'].dt.month >= 6) & (intrepid_data['date'].dt.month <= 9)) & (
        (intrepid_data['date'].dt.weekday >= 0) &
        (intrepid_data['date'].dt.weekday <= 5)) & \
          ((intrepid_data['date'].dt.hour >= 8) & (intrepid_data['date'].dt.hour < 22))

# intermediate Demand_Rate is set for the entire column,
# and other Demand_Rates overwrite it when the conditions are met
intrepid_data.loc[intermediate_peak,"demand_rate1"] = 11.48
intrepid_data.loc[intermediate_peak,"demand_rate2"] = 11.48
intrepid_data.loc[on_peak1, "demand_rate1"] = 8.33 
intrepid_data.loc[on_peak2, "demand_rate2"] = 15.56
# finding cost for all data points, then assigning categories

intrepid_data.loc[intrepid_data["demand_rate1"] == 8.33, "demand_rate_category1"] = "on_peak1"
intrepid_data.loc[intrepid_data["demand_rate2"] == 15.56, "demand_rate_category2"] = "on_peak2"
intrepid_data.loc[intrepid_data["demand_rate1"] == 11.48, "demand_rate_category1"] = "intermediate"
intrepid_data.loc[intrepid_data["demand_rate2"] == 11.48, "demand_rate_category2"] = "intermediate"
intrepid_data.loc[intrepid_data["demand_rate1"] == 0, "demand_rate_category1"] = "off_peak"
intrepid_data.loc[intrepid_data["demand_rate2"] == 0, "demand_rate_category2"] = "off_peak"


#suggest 250kW/500kWh battery here
power = 250

df2 = pd.DataFrame(columns=['intervals','solar','Ein','Eout','charge_state','system_out'])
test_dates_df = pd.DataFrame(columns=['date','dayofyear','lbmp','demand_rate1','demand_rate_category1','demand_rate2','demand_rate_category2',
                                          'original_building_power_kw','output','energy_rate'])
demand_expenses = list()
energy_expenses = list()

for i in set(intrepid_data.date.dt.month):
    test_dates = intrepid_data.loc[(intrepid_data.date.dt.month == i)]#&(data.date.dt.dayofyear <= (i+1))]
    test_dates = test_dates.reset_index(drop=True)
    df,model = op_models.peak_shave(test_dates, power=power, capacity=power*2, eff=.8, itc=False, project_type='storage only', export=False)
    df2 = pd.concat([df2,df])
    test_dates_df = pd.concat([test_dates_df,test_dates])
    demand_expenses.append((model.demand_expenses()))
    energy_expenses.append((model.energy_expenses()))

df2 = df2.reset_index(drop=True)

intrepid_data['Ein'] = df2.Ein
intrepid_data['Eout'] = df2.Eout
intrepid_data['charge_state'] = df2.charge_state
intrepid_data['system_out'] = df2.system_out

demand_expenses_total = sum(demand_expenses)
energy_expenses_total = sum(energy_expenses)
expenses = demand_expenses_total + energy_expenses_total
battery_cost = power*2*340*0.7*0.8

print('Yearly Expenses with no energy storage system: $152095.67')
print('Yearly savings with storage system of size',power,': $',152095.67-expenses)
print('Years to payback: ', battery_cost/(152095.67-expenses))

graph_beginning = 19199
graph_end = 19294
plt.figure(figsize=(15,5))
plt.plot(intrepid_data.index,intrepid_data.original_building_power_kw,intrepid_data.system_out)
plt.figure(figsize=(15,5))
plt.plot(intrepid_data.index[graph_beginning:graph_end],intrepid_data.original_building_power_kw[graph_beginning:graph_end],intrepid_data.system_out[graph_beginning:graph_end])
plt.figure(figsize=(15,5))
plt.plot(intrepid_data.index[graph_beginning:graph_end],intrepid_data.system_out[graph_beginning:graph_end])

#print(intrepid_data.loc[intrepid_data.demand_rate_category1=='off_peak'].system_out.max())
#print(intrepid_data.loc[intrepid_data.demand_rate_category1=='on_peak1'].system_out.max())
#print(intrepid_data.loc[intrepid_data.demand_rate_category2=='on_peak2'].system_out.max())
#print(intrepid_data.loc[intrepid_data.demand_rate_category1=='intermediate'].system_out.max())


#intrepid_data['lbmp'] = total_dayahead_lbmp.loc[(intrepid_data.Timestamp.dt.hour == total_dayahead_lbmp.date.dt.hour)&
#        (intrepid_data.Timestamp.dt.day == total_dayahead_lbmp.date.dt.day)&(intrepid_data.Timestamp.dt.month == total_dayahead_lbmp.date.dt.month)]
    
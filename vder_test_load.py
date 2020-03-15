# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 11:50:53 2019

@author: Richard_Fu
"""

import helper
import op_models
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
data = helper.load_data('vder_test')
start = '2018-11-05 00:00:00'
end = '2019-11-04 23:00:00'
summer = (start,end)
power = 4000
duration = 4
battery_cost=340
incentive=0

def revenue_calc(data,summer,power,duration,battery_cost,incentive,max_line_limit=4000,solar=0,lsrv=True,verbose=False):
    #lsrv_name = 'Event 6'
    #df = df.drop(columns={'day_of_week','lbmp','icap2','drv','community_cred','env_cred'})
    summer_start = pd.to_datetime(summer[0])
    summer_end = pd.to_datetime(summer[1])
    #lsrv_dates = df.loc[(df.date.dt.month==(df.loc[df.lsrv_event==lsrv_name].date.dt.month.reset_index(drop=True)[0])) & (df.date.dt.day==(df.loc[df.lsrv_event==lsrv_name].date.dt.day.reset_index(drop=True)[0]))]
    #lsrv_dates = lsrv_dates.reset_index(drop=True)
    cost1 = 0
    cost2 = 0
    opt1 = 0
    opt2 = 0
    power = power
    capacity = duration*power
    revenue = list()
    new_data_df = pd.DataFrame(columns=['date','intervals','solar','output','Ein','Eout','charge_state','system_out'])
    test_dates_df = pd.DataFrame(columns=['date','day_of_week','solar_output_kw','lbmp','icap','drv','community_cred','env_cred','lsrv_event','lsrv_rate','vder',
                                          'original_building_power','output','energy_rate'])


    if lsrv == False:
        data.loc[data.lsrv_rate != 0, 'lsrv_rate'] = 0
    
    #lsrv_data, lsrv_model = op_models.lsrv(lsrv_dates,power=power, capacity=capacity, max_discharges=1, eff=0.9, project_type='solar+storage', itc=True, max_line_limit=3000)
    
    
    for i in pd.date_range(summer_start,summer_end):
        test_dates = data.loc[(data.date.dt.month == i.month)&(data.date.dt.day == i.day)]#&(data.date.dt.dayofyear <= (i+1))]
        test_dates = test_dates.reset_index(drop=True)
        new_data,model= op_models.lsrv(test_dates, power=power, capacity=capacity, max_discharges=1, eff=.9, project_type='storage only', itc=False, max_line_limit=max_line_limit)
        new_data_df = pd.concat([new_data_df,new_data])
        test_dates_df = pd.concat([test_dates_df,test_dates])
        revenue.append((-model.P()))
#    
#    new_data,model= vder_optimization.vps(data, power=power, capacity=capacity, max_discharges=1, eff=.9, project_type='storage only', itc=True, max_line_limit=max_line_limit)
#    revenue = -model.P()
    new_data_df = new_data_df.reset_index(drop=True)
    test_dates_df = test_dates_df.reset_index(drop=True)
    battery_cost=battery_cost
    incentive=incentive
    #cost1 = (capacity*(550-200)*0.7*0.8)
    cost2 = (capacity*(battery_cost-incentive)*0.74*0.8)
    #opt1 = ((sum(revenue)-822323.16)*10-cost1)
#    if verbose:
#        print('Must do the VDER components calculation with solar only first before doing calculations for storage only.')
#        
#        
#    if power == 0:
#        solar = 0        
    vder_components = sum(revenue) - solar
#
#
#    print("Yearly revenue: ", sum(revenue))
#    #print(opt1)
##    print('10 year revenue storage only: ' ,opt2)    
#    print("VDER component compensation per year : ", vder_components)
#    #print(op1ratio)
#    #print(op2ratio)
#    
#    print("June Total Energy: ", new_data_df.loc[new_data_df.date.dt.month==6].Eout.sum())
#    print("July Total Energy: ", new_data_df.loc[new_data_df.date.dt.month==7].Eout.sum())
#    print("August Total Energy: ", new_data_df.loc[new_data_df.date.dt.month==8].Eout.sum())
#    print("September Total Energy: ", new_data_df.loc[new_data_df.date.dt.month==9].Eout.sum())
#    
#    new_data_df['lbmp'] = test_dates_df.lbmp
#    new_data_df['drv'] = test_dates_df.drv
#    new_data_df['icap'] = test_dates_df.icap
#    new_data_df['community_cred'] = test_dates_df.community_cred
#    new_data_df['env_cred'] = test_dates_df.env_cred
#    new_data_df['vder'] = test_dates_df.vder
#
#
#    print('LBMP portion: ', sum(new_data_df.lbmp*(new_data_df.system_out)))   
#    print('DRV portion: ', sum(new_data_df.drv*(new_data_df.system_out)))
#    print('ICAP2 portion: ', sum(new_data_df.icap*(new_data_df.system_out)))
#    print('Community portion: ', sum(new_data_df.community_cred*(new_data_df.system_out)))
#    print('Env portion: ', sum(new_data_df.env_cred*(new_data_df.system_out)))

    return(vder_components,new_data_df,model)
    
    
#solar_val,new_data_df1 = revenue_calc(data,summer,0,duration,battery_cost,incentive,solar=0,lsrv=False)
vder_components,new_data,model = revenue_calc(data,summer,power,duration,battery_cost,incentive,solar=0,lsrv=True)
vder_components_no_lsrv, new_data_no_lsrv, model_no_lsrv = revenue_calc(data,summer,power,duration,battery_cost,incentive,solar=0,lsrv=False)
new_data['drv'] = data.drv
new_data_no_lsrv['drv'] = data.drv
drv_df = new_data.loc[new_data.drv>0]
drv_df2 = new_data_no_lsrv.loc[new_data_no_lsrv.drv>0]

#we take one drv day to find the annual peak value for ICAP 3 calculations, then we distribute all of our power across that bit
# we take the entire drv day because we do not know when the annual peak will occur, and thus we must always be ready for it
drv_value = 0.2218
one_day_drv = 10*power*duration*drv_value
#for annual icap it depends on power (will be using Nov 2018-Oct 2019 values because they are most recent)
icap3 = [3.04,1.89,1.74,1.65,1.64,1.65,4.36,5.52,5.45,5.14,4.96,4.95] 
icap3_revenue = sum(icap3)*power
total_revenue = vder_components + icap3_revenue #lsrv assumes no icap3 revenue
total_revenue_no_lsrv = vder_components_no_lsrv + icap3_revenue


#sample = new_data_df[581:645]
#sample2 = new_data_df[600:624].reset_index(drop=True)
#plt.step(sample2.intervals,sample2.Ein)



#plt.figure(figsize=(15,5))
#plt.step(sample2.date.dt.hour,sample2.system_out, label='Total Output',where='post')
#plt.step(sample2.date.dt.hour,sample2.charge_state, label='State of Charge (total kWh)',where='post')
#plt.step(sample2.date.dt.hour,sample2.solar, label='Solar Output',where='post')
#plt.step(sample2.date.dt.hour,sample2.Eout, label='Battery discharging',where='post')
#plt.legend()
#plt.xticks(np.arange(0,24, step=1))
#plt.title('3.375MW inverter proposed system 1 day operation')
#plt.xlabel('Hours')
#plt.ylabel('Power (kW)')
#plt.grid()
#plt.axvspan(14,19,facecolor='grey',alpha=0.25)
#plt.axvspan(14,18,facecolor='grey',alpha=0.4)
#plt.savefig('3MWlinelimit_IBM.png')


#plt.figure(figsize=(15,5))
#plt.step(test_dates_df1.index,test_dates_df1.lbmp, label='LBMP Contribution')
#plt.step(test_dates_df1.index,test_dates_df1.icap2, label='ICAP2 Contribution')
#plt.step(test_dates_df1.index,test_dates_df1.drv, label='DRV Contribution')
#plt.step(test_dates_df1.index,test_dates_df1.vder, label='Total VDER')
#plt.legend()
#plt.xticks(np.arange(0,24, step=1))
#plt.title('Contribution to VDER storage income')
#plt.xlabel('Hours')
#plt.ylabel('Income ($/kWh)')
#
#plt.figure()
#plt.plot(lsrv_data.index,lsrv_data.Eout+lsrv_data.solar-lsrv_data.Ein)
#plt.plot(lsrv_data.index,lsrv_data.Eout)
#plt.xticks(np.arange(0, 24, step=1))
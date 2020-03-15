# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 10:20:27 2019

@author: IST_1
"""

import pandas as pd
from op_models import vder_only
import numpy as np
from matplotlib import pyplot as plt

df = pd.read_csv('ERCOT Day-Ahead Price Energy.csv')
anc = pd.read_csv('ERCOT Ancillary Services MCP.csv')

df.Date = pd.to_datetime(df.Date)
anc.Date = pd.to_datetime(anc.Date)


zones = list(set(df.zone))

houston_lmp = df.loc[df.zone=='LZ_HOUSTON'].reset_index(drop=True)
west_lmp = df.loc[df.zone=='LZ_WEST'].reset_index(drop=True)
south_lmp = df.loc[df.zone=='LZ_SOUTH'].reset_index(drop=True)
aen_lmp = df.loc[df.zone=='LZ_AEN'].reset_index(drop=True)
lcra_lmp = df.loc[df.zone=='LZ_LCRA'].reset_index(drop=True)
cps_lmp = df.loc[df.zone=='LZ_CPS'].reset_index(drop=True)
north_lmp = df.loc[df.zone=='LZ_NORTH'].reset_index(drop=True)
raybn_lmp = df.loc[df.zone=='LZ_RAYBN'].reset_index(drop=True)


reg_up = anc.loc[anc.ancillarytype=='Regulation Up'].reset_index(drop=True)
reg_up = reg_up.rename(columns={'mcpc':'Regulation Up'})
reg_up = reg_up.drop(columns={'ancillarytype'})

reg_reserve = anc.loc[anc.ancillarytype=='Responsive Reserve Service'].reset_index(drop=True)
reg_reserve = reg_reserve.rename(columns={'mcpc':'Regulation Reserve Service'})
reg_reserve = reg_reserve.drop(columns={'ancillarytype'})

reg_down = anc.loc[anc.ancillarytype=='Regulation Down'].reset_index(drop=True)
reg_down = reg_down.rename(columns={'mcpc':'Regulation Down'})
reg_down = reg_down.drop(columns={'ancillarytype'})

nonspin = anc.loc[anc.ancillarytype=='Non Spin'].reset_index(drop=True)
nonspin = nonspin.rename(columns={'mcpc':'Non Spin'})
nonspin = nonspin.drop(columns={'ancillarytype'})

ancillary = reg_up
ancillary['Responsive Reserve Service'] = reg_reserve['Regulation Reserve Service']
ancillary['Regulation Down'] = reg_down['Regulation Down']
ancillary['Non Spin'] = nonspin['Non Spin']

merged_1 = pd.merge(houston_lmp, ancillary, on=['Date'], how='inner')



list_of_regions = [houston_lmp,south_lmp,aen_lmp,lcra_lmp,cps_lmp,north_lmp,raybn_lmp,west_lmp]
list_of_regions2 = list()

for region in list_of_regions:
    region = pd.merge(region,ancillary,on=['Date'], how='inner')
    list_of_regions2.append(region)
    


df2 = pd.concat(list_of_regions2).sort_values(by='Date')

df2.to_csv('ERCOT Ancillary and Energy Pricing.csv')

#for region in list_of_regions:
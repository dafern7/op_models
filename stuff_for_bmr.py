# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 14:51:42 2019

@author: IST_1
"""
import helper
import op_models
import baseload_mod

df = helper.pv_data(address='Bahamas', solar_size_kw_dc=5000)
solar = df[1:500]
capacity=20000
power=10000
diesel_size=10000
capacity_cost=550

output, model = baseload_mod.base_load(solar,capacity,power,diesel_size,capacity_cost,strict_baseload=2000,relaxed_baseload=0,window=(0,24),relaxed_window=(0,0),eff=.8)
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 14:53:02 2019

@author: IST_1
"""

import numpy as np
from pyomo.environ import *
import pandas as pd
import datetime
import helper
from op_helpers import model_to_df, filter_demand_rates, filter_data, filter_lsrv_df


def base_load(df,capacity,power,diesel_size,capacity_cost,eff=.8):
    """
    Optimize the charge/discharge behavior of a battery storage unit over a
    full year. Assume perfect foresight of electricity prices. The battery
    has a discharge constraint equal to its storage capacity and round-trip
    efficiency of 80%.

    Parameters
    ----------


    :param df : solar dataframe with hourly data
    :param power: power of the battery in kw
    :param capacity: capacity of the battery in kwh
    :param eff: optional, round trip efficiency
    :param compensation_rate: $/kWh for energy injected to the grid
    :param reserve_rate: $/kWh for energy stored
    :param base_load_window: range of hours of the base load, default (0,23)

    :return dataframe
        hourly state of charge, charge/discharge rates
    """
    # assertions
    assert 'solar_output_kw' in df.columns, 'no solar_output_kw column'
    assert 'hour' in df.columns, 'no hour column in the dataframe'
    #assert isinstance(base_load_window, tuple), 'base_load_window must be tuple'

    # variable for converting power to energy (since this is an hour by hour dispatch, dih =1)
    dih = 1
    # index must start with 0
    if df.index[0] != 0:
        df.reset_index(drop=True, inplace=True)

    # Define model sets and parameters
    model = ConcreteModel()
    model.T = Set(doc='index', initialize=(df.index + 1).tolist(), ordered=True)
    model.DieselSize = Param(initialize=diesel_size)
    model.max_cap = Param(initialize=capacity)
    model.max_power = Param(initialize=power)
    
    # create solar vector
    solar_dic = dict(zip(model.T.keys(), df.solar_output_kw.tolist()))
    model.Solar = Param(model.T, initialize=solar_dic, doc='solar output')

    # variables: Charge, discharge, binary variable, state of charge, Pmax
    model.Ein = Var(model.T, domain=NonNegativeReals, initialize=0.0)
    model.Eout = Var(model.T, domain=NonNegativeReals, initialize=0.0)
    model.Smax = Var(bounds=(0,model.max_cap),initialize=0.0)
    model.Rmax = Var(bounds=(0,model.max_power),initialize=0.0)
    model.D = Var(model.T, bounds=(0,model.DieselSize), initialize=0.0)
    model.X = Var(model.T, domain=Binary, initialize=0.0)
    model.S = Var(model.T, domain=NonNegativeReals, initialize=0.0)
    
#    def storage_constraint(model):
#        return model.Smax == 4*model.Rmax
#    model.storage_constraint = Constraint(rule=storage_constraint)

    # charge the battery from solar power only
    def only_solar_constraint(model, t):
        return model.Ein[t] <= model.Solar[t] + model.D[t]
    model.only_solar_constraint = Constraint(model.T, rule=only_solar_constraint)

    
#    def diesel_constraint(model, t):
#        return sum(model.D[t] for t in model.T) <= 0.1*sum(model.Eout[t] + model.Solar[t] - model.Ein[t] + model.D[t] for t in model.T)
#    model.diesel_constraint = Constraint(model.T, rule=diesel_constraint)


    # Storage State
    def storage_state(model, t):
        'Storage changes with flows in/out and efficiency losses'
        # Set first hour state of charge to half of max
        if t == model.T.first():
            return model.S[t] == 0
        else:
            return (model.S[t] == (model.S[t - 1]
                                   + dih * (model.Ein[t-1] * np.sqrt(eff))
                                   - dih * (model.Eout[t-1] / np.sqrt(eff))))

    model.charge_state = Constraint(model.T, rule=storage_state)

#    def base_load_constraint1(model, t):
#        index = t-1
#        if df.iloc[index].hour in list(range(8, 11)):
#            return model.D[t] + model.Solar[t] + model.Eout[t] - model.Ein[t] == model.D[t+1] + model.Solar[t+1] + model.Eout[t+1] - model.Ein[t+1]
#        elif df.iloc[index].hour in list(range(11, 2)):
#            return model.D[t] + model.Solar[t] + model.Eout[t] - model.Ein[t] == model.D[t+1] + model.Solar[t+1] + model.Eout[t+1] - model.Ein[t+1]
#        elif df.iloc[index].hour in list(range(2, 6)):  
#            return model.D[t] + model.Solar[t] + model.Eout[t] - model.Ein[t] == model.D[t+1] + model.Solar[t+1] + model.Eout[t+1] - model.Ein[t+1]
#        else:
#            return Constraint.Skip
#            #return model.D[t] + model.Solar[t] + model.Eout[t] - model.Ein[t] == 0
        
#    model.base_load_constraint1 = Constraint(model.T, rule=base_load_constraint1)
    
    def base_load_constraint1(model, t):
        index= t-1
        window = list(range(0, 24))
        if df.iloc[index].hour in list(range(8, 12)):
            return (model.D[t] + model.Solar[t] + model.Eout[t] - model.Ein[t]) == 2000
        elif df.iloc[index].hour == window[-1]+1:
            return Constraint.Skip
        else:
            return (model.D[t] + model.Solar[t] + model.Eout[t] - model.Ein[t]) == 2000
    
    model.base_load_constraint1 = Constraint(model.T, rule=base_load_constraint1)


    def discharge_constraint(model, t):
        """Maximum discharge within a single hour"""
        return model.Eout[t] <= model.Rmax * model.X[t]
    model.discharge = Constraint(model.T, rule=discharge_constraint)

    def charge_constraint(model, t):
        """Maximum charge within a single hour"""
        return model.Ein[t] <= model.Rmax * (1 - model.X[t])
    model.charge = Constraint(model.T, rule=charge_constraint)

    # Without a constraint the model would discharge in the final hour
    # even when SOC was 0.
    def positive_charge(model, t):
        'Limit discharge to the amount of charge in battery, including losses'
        return model.Eout[t] <= model.S[t]*np.sqrt(eff) / dih
    model.positive_charge = Constraint(model.T, rule=positive_charge)
    
    def too_much_storage(model, t):
        return model.S[t] <= model.Smax   
    model.too_much_storage = Constraint(model.T, rule=too_much_storage)

    # Define the battery income
    # Income:
    
    cost = model.Smax*capacity_cost + 10*sum(model.D[t] *.11 for t in model.T)
    
    model.cost = cost
    model.objective = Objective(expr=cost, sense=minimize)

    # Solve the model
    solver = SolverFactory('gurobi')
    solver.solve(model)

    return model_to_df(model,'solar+storage'), model
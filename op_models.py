import numpy as np
from pyomo.environ import *
import pandas as pd
import datetime
from op_helpers import model_to_df, filter_demand_rates, filter_data, filter_lsrv_df


def base_load(df, power, capacity, eff=.8, compensation_rate=0, reserve_rate=0, base_load_window=(0,23)):
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
    assert isinstance(base_load_window, tuple), 'base_load_window must be tuple'

    # variable for converting power to energy (since this is an hour by hour dispatch, dih =1)
    dih = 1
    # index must start with 0
    if df.index[0] != 0:
        df.reset_index(drop=True, inplace=True)

    # Define model sets and parameters
    model = ConcreteModel()
    model.T = Set(doc='index', initialize=(df.index + 1).tolist(), ordered=True)
    model.Rmax = Param(initialize=power, doc='Max rate of power flow (kW) in or out')
    model.Smax = Param(initialize=capacity, doc='Max storage (kWh)')
    # create solar vector
    solar_dic = dict(zip(model.T.keys(), df.solar_output_kw.tolist()))
    model.Solar = Param(model.T, initialize=solar_dic, doc='solar output')

    # variables: Charge, discharge, binary variable, state of charge, Pmax
    model.Ein = Var(model.T, domain=NonNegativeReals, initialize=0.0)
    model.Eout = Var(model.T, domain=NonNegativeReals, initialize=0.0)
    model.X = Var(model.T, domain=Binary, initialize=0.0)
    model.S = Var(model.T, bounds=(0, model.Smax), initialize=model.Smax)

    # charge the battery from solar power only
    def only_solar_constraint(model, t):
        return model.Ein[t] <= model.Solar[t]
    model.only_solar_constraint = Constraint(model.T, rule=only_solar_constraint)

    # Pmax Constraint, not active
    # TODO: Ask if there are any power constraints for the project
    def power_constraint(model, t):
        rate_cat = df.demand_rate_category.iloc[t - 1]
        return (model.Output[t] + model.Ein[t] - model.Eout[t]) <= model.Pmax[rate_cat]

    #model.power_constraint = Constraint(model.T, rule=power_constraint)

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

    def base_load_constraint(model, t):
        index = t-1
        window = list(range(base_load_window[0], base_load_window[1]))
        if df.iloc[index].hour in window:
            return model.Solar[t] + model.Eout[t] - model.Ein[t] == model.Solar[t+1] + model.Eout[t+1] - model.Ein[t+1]
        elif df.iloc[index].hour == window[-1]+1:
            return Constraint.Skip
        else:
            return model.Solar[t] + model.Eout[t] - model.Ein[t] == 0

    model.base_load_constraint = Constraint(model.T, rule=base_load_constraint)

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

    # Define the battery income
    # Income:
    energy_income = sum(compensation_rate * (model.Eout[t] + model.Solar[t] - model.Ein[t]) for t in model.T)
    reserve_income = sum(reserve_rate * model.S[t] for t in model.T)

    income = energy_income + reserve_income
    model.P = income
    model.objective = Objective(expr=income, sense=maximize)

    # Solve the model
    solver = SolverFactory('gurobi')
    solver.solve(model)

    return model_to_df(model), model


def peak_shave(df, power, capacity, eff=.8, itc=False, project_type='solar+storage', export=False):
    assert 'date' in df.columns, 'dataframe has no date column'
    assert isinstance(df.date.iloc[0], datetime.date), 'date is not a datetime data type'

    # index must start with 0
    df.reset_index(drop=True, inplace=True)

    # dih: data interval hours
    dih = (df.date.iloc[1] - df.date.iloc[0])
    dih = dih.total_seconds() / 3600
    dih = abs(dih)
    print(dih)
    num_col = 0
    
    for colum in df.columns: 
        if 'demand_rate_category' in colum:
            num_col = num_col+1
            
    demand_categories, demand_rates_dic = filter_demand_rates(df,num_columns=num_col)

    model = ConcreteModel()
    model.T = Set(doc='index', initialize=(df.index + 1).tolist(), ordered=True)
    model.col = Set(initialize=list(range(1,num_col+1)),ordered=True)
    model.Rmax = Param(initialize=power, doc='Max rate of power flow (kW) in or out')
    model.DemandCategories = Set(initialize=demand_categories, doc='demand rate categories', ordered=True)
    model.Smax = Param(initialize=capacity, doc='Max storage (kWh)')
    model.date = Set(doc='date', initialize=df.date ,ordered=True)
    # create solar vector
    if project_type == 'solar+storage':
        solar_dic = dict(zip(model.T.keys(), df.solar_output_kw.tolist()))
        model.Solar = Param(model.T, initialize=solar_dic, doc='solar output')
        df['output'] = df.power_post_solar_kw
    else:
        df['output'] = df.original_building_power_kw
    # output vector (without storage)
    model.Output = Param(model.T, initialize=dict(zip(model.T.keys(), df.output.tolist())),
                         doc='original building power or post solar')

    # Rates

    model.Erate = Param(model.T, initialize=dict(zip(list(model.T), df.energy_rate.tolist())), doc='ENERGY RATES')
    model.DemandRates = Param(model.DemandCategories, initialize=demand_rates_dic)

    # variables: Charge, discharge, binary variable, state of charge, Pmax
    model.Ein = Var(model.T, domain=NonNegativeReals, initialize=100.0)
    model.Eout = Var(model.T, domain=NonNegativeReals, initialize=100.0)
    model.X = Var(model.T, domain=Binary, initialize=0.0)
    model.S = Var(model.T, bounds=(0, model.Smax), initialize=model.Smax)
    model.Pmax = Var(model.DemandCategories, domain=NonNegativeReals, initialize=0)

    if itc:
        assert 'solar_output_kw' in df.columns, "can't add ITC without solar"
        model.SolarOut = Param(model.T, initialize=dict(zip(model.T.keys(), df.solar_output_kw.tolist())), doc='solar')

        def itc_constraint(model, t):
            return model.Ein[t] <= model.SolarOut[t]
        model.itc_constraint = Constraint(model.T, rule=itc_constraint)

    if not export:
        def no_export(model, t):
            return model.Output[t] + model.Ein[t] - model.Eout[t] >= 0

        model.no_export = Constraint(model.T, rule=no_export)
    
#    def output_power_constraint(model, t):
#    
#        return model.Ein[t]<=model.Output[t]
#
#    model.output_power_constraint = Constraint(model.T, rule=output_power_constraint)

    # Pmax Constraint
    if len(model.col)==1:
        def power_constraint(model, t):
            rate_cat = df.demand_rate_category.iloc[t - 1]
            return (model.Output[t] + model.Ein[t] - model.Eout[t]) <= model.Pmax[rate_cat]

        model.power_constraint = Constraint(model.T, rule=power_constraint)
        
    else:
        def column_constraint(model,t,col):
            cat_col = df['demand_rate_category'+str(col)]
            rate_cat = cat_col.iloc[t-1]
            #print(model.Output[t]+model.Ein[t].value-model.Eout[t].value)
            #print(model.Pmax[rate_cat].value)
            return(model.Output[t]+model.Ein[t]-model.Eout[t] <= model.Pmax[rate_cat])

        model.column_constraint = Constraint(model.T,model.col,rule=column_constraint)
    

    # Storage State
    def storage_state(model, t):
        'Storage changes with flows in/out and efficiency losses'
        # Set first hour state of charge to half of max
        if t == model.T.first():
            return model.S[t] == 0
        else:
            return (model.S[t] == (model.S[t - 1]
                                   + dih * (model.Ein[t - 1] * np.sqrt(eff))
                                   - dih * (model.Eout[t - 1] / np.sqrt(eff))))

    model.charge_state = Constraint(model.T, rule=storage_state)

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
        return model.Eout[t] <= model.S[t]*np.sqrt(eff) /dih
    model.positive_charge = Constraint(model.T, rule=positive_charge)

    energy_expenses = dih * sum(
        model.Erate[t] * (model.Output[t] + model.Ein[t] - model.Eout[t]) for t in model.T)
    model.energy_expenses = energy_expenses
    demand_expenses = sum(model.Pmax[key] * model.DemandRates[key] for key in model.Pmax.keys())
    model.demand_expenses = demand_expenses
    expenses = energy_expenses + demand_expenses
    model.expenses = expenses
    model.objective = Objective(expr=expenses, sense=minimize)

    # Solve the model
    solver = SolverFactory('gurobi')
    solver.solve(model)
    
    
    intervals = range(model.T.first(), model.T.last() + 1)
    Ein = [model.Ein[i].value for i in intervals]
    Eout = [model.Eout[i].value for i in intervals]
    date = [model.date[i] for i in intervals]
    charge_state = [model.S[i].value for i in intervals]
    output = [model.Output[i] for i in intervals]
    if project_type == 'solar+storage':
        solar = [model.Solar[i] for i in intervals]
    else:
        solar = [0 for i in intervals]
    system_out = [(-model.Eout[i].value + model.Output[i] + model.Ein[i].value) for i in intervals]
    df_dict = dict(
        intervals=intervals,
        solar=solar,
        date=date,
        output=output,
        Ein=Ein,
        Eout=Eout,
        charge_state=charge_state,
        system_out=system_out,
    )

    df = pd.DataFrame(df_dict)


    return df, model

def microgrid(df,solar,eff=.8,dod=0.85,power_val=20,capacity_val=800):
    assert 'date' in df.columns, 'dataframe has no date column'
    assert isinstance(df.date.iloc[0], datetime.date), 'date is not a datetime data type'

    # index must start with 0
    df.reset_index(drop=True, inplace=True)

    # dih: data interval hours
    dih = (df.date.iloc[1] - df.date.iloc[0])
    dih = dih.total_seconds() / 3600
    dih = abs(dih)
    large_system_cost=1000000

    
    model = ConcreteModel()
    model.T = Set(doc='index', initialize=(df.index + 1).tolist(), ordered=True)
    
    solar_dic = dict(zip(model.T.keys(), solar.solar_output_kw.tolist()))
    model.Solar = Param(model.T, initialize=solar_dic, doc='solar output')
    building_power_dic = dict(zip(model.T.keys(), df.original_building_power_kw.tolist()))
    model.Building_Power = Param(model.T, initialize=building_power_dic, doc='original building power')
    #df['output'] = df.power_post_solar_kw
    
    model.Output = Param(model.T, initialize=dict(zip(model.T.keys(), df.original_building_power_kw.tolist())),
                         doc='original building power')
    #model.Smax=Param(model.T,initialize=20000,doc='max capacity')
    
    
    model.Ein = Var(model.T, domain=NonNegativeReals, initialize=0.0)
    model.Eout = Var(model.T, domain=NonNegativeReals, initialize=0.0)
    model.X = Var(model.T, domain=Binary, initialize=0.0)
    model.Smax = Var(domain=NonNegativeReals, initialize=0.0)
    model.Rmax = Var(domain=NonNegativeReals, initialize=0.0)
    model.alpha = Var(bounds=(0,1), initialize = 0.5)        
    model.S = Var(model.T, domain=NonNegativeReals, initialize=0.0)
    
    def power_constraint(model, t):
        return ((model.Building_Power[t] - model.Solar[t]*model.alpha + model.Ein[t] - model.Eout[t]) == 0)
    model.power_constraint = Constraint(model.T, rule=power_constraint)

    def solar_constraint(model, t):
        return (model.Ein[t] <= model.Solar[t]*model.alpha)
    model.solar_constraint = Constraint(model.T, rule=solar_constraint)
               
    def storage_state(model, t):
        'Storage changes with flows in/out and efficiency losses'
        # Set first hour state of charge to half of max
        if t == model.T.first():
            return model.S[t] == 0
        else:
            return (model.S[t] == (model.S[t - 1]
                                   + dih * (model.Ein[t - 1] * np.sqrt(eff))
                                   - dih * (model.Eout[t - 1] / np.sqrt(eff))))
    model.charge_state = Constraint(model.T, rule=storage_state) 
    
    def discharge_constraint(model, t):
        """Maximum discharge within a single hour"""
        return model.Eout[t] <= model.Rmax
    model.discharge = Constraint(model.T, rule=discharge_constraint)

    def charge_constraint(model, t):
        """Maximum charge within a single hour"""
        return model.Ein[t] <= model.Rmax
    model.charge = Constraint(model.T, rule=charge_constraint) 
    
    def positive_charge(model, t):
        'Limit discharge to the amount of charge in battery, including losses'
        return model.Eout[t] <= model.S[t]*np.sqrt(eff) / dih
    model.positive_charge = Constraint(model.T, rule=positive_charge)
        
    def binary_constraint_in(model,t):
        return model.Ein[t] <= model.X[t]    
    model.binary_constraint1 = Constraint(model.T, rule=binary_constraint_in)
    
    def binary_constraint_out(model,t):
        return model.Eout[t] <= (1-model.X[t])*1 # TODO this is wrong and must be revised
    model.binary_constraint2 = Constraint(model.T, rule=binary_constraint_out)
    
    def soc_constraint(model,t):
        return model.S[t] <= model.Smax*dod    
    model.soc = Constraint(model.T, rule=soc_constraint)

    
    expenses = capacity_val*model.Smax + power_val*model.Rmax + large_system_cost*model.alpha
    model.objective = Objective(expr=expenses, sense=minimize)

    # Solve the model
    solver = SolverFactory('gurobi')
    solver.solve(model)

    return model_to_df(model,'solar+storage'), model


def lsrv(df, power, capacity, max_discharges, eff=.9, project_type='storage only', itc=False, max_line_limit=5000):
    """
    Optimize the charge/discharge behavior of a battery storage unit over a
    full year. Assume perfect foresight of electricity prices. The battery
    has a discharge constraint equal to its storage capacity and round-trip
    efficiency of 80%.

    Parameters
    ----------

    :param df : dataframe
        TODO: CHANGE THIS
        dataframe, Note: each demand rate category MUST have 1 demand rate only. If there is more than one rate for
        each rate category, it will return an error.
    :param power: power of the battery in kw
    :param capacity: capacity of the battery in kwh
    :param max_discharges: maximum discharge in 24 hour window
    :param eff: optional, round trip efficiency
    :param project_type: 'solar+storage' or 'solar+storage'
    :param itc: set True if ITC is desired

    :return dataframe
        hourly state of charge, charge/discharge behavior, lbmp, and time stamp
    """

    # Filter the data
    df = filter_data(df, project_type)

    # dih: data interval hours
    dih = (df.date.iloc[1] - df.date.iloc[0])
    dih = dih.total_seconds() / 3600
    dih = abs(dih)

    # TODO check back this function
    lsrv_events, lsrv_rates_dic = filter_lsrv_df(df)

    # Define model sets and parameters
    model = ConcreteModel()
    model.T = Set(doc='index', initialize=(df.index + 1).tolist(), ordered=True)
    model.date = Set(doc='date', initialize=df.date ,ordered=True)
    model.LSRV_Events = Set(initialize=lsrv_events, doc='LSRV Events', ordered=True)
    model.Rmax = Param(initialize=power, doc='Max rate of power flow (kW) in or out')
    model.Smax = Param(initialize=(capacity/np.sqrt(eff)), doc='Max storage (kWh), scaled for efficiency')
    model.Dmax = Param(initialize=capacity * max_discharges, doc='Max discharge in 24 hour period')
    model.Line_Max = Param(initialize=-max_line_limit, doc='maximum line power')
    model.Output = Param(model.T, initialize=dict(zip(model.T.keys(), df.output.tolist())),
                         doc='original building power or post solar')

    # add energy rate column in case the project is not billed for energy (ex: solar+storage)
    if 'energy_rate' not in df.columns:
        df['energy_rate'] = 0
    model.Erate = Param(model.T, initialize=dict(zip(list(model.T), df.energy_rate.tolist())), doc='ENERGY RATES')

    model.LSRV_Rates = Param(model.LSRV_Events, initialize=lsrv_rates_dic)
    model.Vder = Param(model.T, initialize=dict(zip(list(model.T), df.vder.tolist())), doc='VDER RATES')

    # variables: Charge, discharge, binary variable, state of charge, Pmax
    model.Ein = Var(model.T, domain=NonNegativeReals, initialize=0.0)
    model.Eout = Var(model.T, domain=NonNegativeReals, initialize=0.0)
    model.X = Var(model.T, domain=Binary, initialize=0.0)
    model.S = Var(model.T, bounds=(0, model.Smax), initialize=model.Smax)
    model.P_LSRV = Var(model.LSRV_Events, domain=NonNegativeReals, initialize=0)

    # model.Qnet = summation(model.Output + model.Ein - model.Eout)

    # set itc constraint
    if project_type == 'solar+storage':
        assert 'solar_output_kw' in df.columns, "can't add ITC without solar"
        model.Solar = Param(model.T, initialize=dict(zip(model.T.keys(), df.solar_output_kw.tolist())), doc='solar')

        def itc_constraint(model, t):
            return model.Ein[t] <= model.Solar[t]
        model.itc_constraint = Constraint(model.T, rule=itc_constraint)
        
#             Pmax Constraint
#     TODO make sure signs are correct
        
        def power_constraint(model, t):
            event = df.lsrv_event.iloc[t - 1]
            return (-model.Output[t] - model.Ein[t] + model.Eout[t]) >= model.P_LSRV[event]
    
        model.power_constraint = Constraint(model.T, rule=power_constraint)
        
    elif project_type =='storage only':
        
        def lsrv_constraint(model, t):
            event = df.lsrv_event.iloc[t - 1]
            return model.Eout[t]>=model.P_LSRV[event]
        model.lsrv_constraint = Constraint(model.T, rule=lsrv_constraint)
        


    # Line limit:
    def line_limit(model, t):
        return (model.Output[t] + model.Ein[t] - model.Eout[t]) >= model.Line_Max

    model.line_limit = Constraint(model.T, rule=line_limit)


    # Storage State
    def storage_state(model, t):
        'Storage changes with flows in/out and efficiency losses'
        # Set first hour state of charge to half of max
        if t == model.T.first():
            return model.S[t] == 0
        else:
            return (model.S[t] == (model.S[t - 1]
                                   + dih * (model.Ein[t - 1] * np.sqrt(eff))
                                   - dih * (model.Eout[t - 1] / np.sqrt(eff))))

    model.charge_state = Constraint(model.T, rule=storage_state)

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
        return model.Eout[t] <= model.S[t] * np.sqrt(eff) / dih

    model.positive_charge = Constraint(model.T, rule=positive_charge)

    # TODO make sure  there are no limits on daily discharge limit
    def daily_discharge_limit(model, t):
        """Limit on discharge within a 24 hour period"""
        max_t = model.T.last()
        num_intervals = int(24/dih)
        # Check all t until the last 24 hours
        # No need to check with < 24 hours remaining because the constraint is
        # already in place for a larger number of hours
        if t < max_t - num_intervals + 2:
            return dih * sum(model.Eout[i] for i in range(t, t + num_intervals)) <= model.Dmax
        else:
            return Constraint.Skip

    model.limit_out = Constraint(model.T, rule=daily_discharge_limit)

    # Define the battery income, expenses, and profit
    # Income: (VDER income must be multiplied by dih since the compensation is $/kwh)
    vder_income = sum(model.Vder[t] * (model.Eout[t]) for t in model.T) * dih
    lsrv_income = sum(model.P_LSRV[key] * model.LSRV_Rates[key] for key in model.P_LSRV.keys())
    model.lsrv_income = lsrv_income
    income = vder_income + lsrv_income
    expenses = dih * sum(model.Vder[t] * (model.Ein[t]) for t in model.T)
    cost = expenses - income
    model.P = cost
    model.objective = Objective(expr=cost, sense=minimize)

    # Solve the model
    solver = SolverFactory('gurobi')
    solver.solve(model)
    
    
    intervals = range(model.T.first(), model.T.last() + 1)
    date = [model.date[i] for i in intervals]
    Ein = [model.Ein[i].value for i in intervals]
    date = [model.date[i] for i in intervals]
    Eout = [model.Eout[i].value for i in intervals]
    charge_state = [model.S[i].value for i in intervals]
    output = [model.Output[i] for i in intervals]
    if project_type == 'solar+storage':
        solar = [model.Solar[i] for i in intervals]
    else:
        solar = [0 for i in intervals]
    system_out = [(model.Eout[i].value + model.Output[i] - model.Ein[i].value) for i in intervals]
    df_dict = dict(
        intervals=intervals,
        solar=solar,
        output=output,
        date = date,
        Ein=Ein,
        Eout=Eout,
        charge_state=charge_state,
        system_out=system_out,
    )

    df = pd.DataFrame(df_dict)

    return df, model

def vder_only(df, power, capacity, max_discharges, eff=.9, project_type='storage only', itc=False, max_line_limit=5000):
    """
    Optimize the charge/discharge behavior of a battery storage unit over a
    full year. Assume perfect foresight of electricity prices. The battery
    has a discharge constraint equal to its storage capacity and round-trip
    efficiency of 80%.

    Parameters
    ----------

    :param df : dataframe
        TODO: CHANGE THIS
        dataframe, Note: each demand rate category MUST have 1 demand rate only. If there is more than one rate for
        each rate category, it will return an error.
    :param power: power of the battery in kw
    :param capacity: capacity of the battery in kwh
    :param max_discharges: maximum discharge in 24 hour window
    :param eff: optional, round trip efficiency
    :param project_type: 'solar+storage' or 'solar+storage'
    :param itc: set True if ITC is desired

    :return dataframe
        hourly state of charge, charge/discharge behavior, lbmp, and time stamp
    """

    # Filter the data
    df = filter_data(df, project_type)

    # dih: data interval hours
    dih = (df.date.iloc[1] - df.date.iloc[0])
    dih = dih.total_seconds() / 3600
    dih = abs(dih)


    # Define model sets and parameters
    model = ConcreteModel()
    model.T = Set(doc='index', initialize=(df.index + 1).tolist(), ordered=True)
    model.date = Set(doc='date', initialize=df.date ,ordered=True)
    model.Rmax = Param(initialize=power, doc='Max rate of power flow (kW) in or out')
    model.Smax = Param(initialize=(capacity/np.sqrt(eff)), doc='Max storage (kWh), scaled for efficiency')
    model.Dmax = Param(initialize=capacity * max_discharges, doc='Max discharge in 24 hour period')
    model.Line_Max = Param(initialize=-max_line_limit, doc='maximum line power')
    model.Output = Param(model.T, initialize=dict(zip(model.T.keys(), df.output.tolist())),
                         doc='original building power or post solar')


    model.Vder = Param(model.T, initialize=dict(zip(list(model.T), df.vder.tolist())), doc='VDER RATES')

    # variables: Charge, discharge, binary variable, state of charge, Pmax
    model.Ein = Var(model.T, domain=NonNegativeReals, initialize=0.0)
    model.Eout = Var(model.T, domain=NonNegativeReals, initialize=0.0)
    model.X = Var(model.T, domain=Binary, initialize=0.0)
    model.S = Var(model.T, bounds=(0, model.Smax), initialize=model.Smax)


    # model.Qnet = summation(model.Output + model.Ein - model.Eout)

    # set itc constraint
    if project_type == 'solar+storage':
        assert 'solar_output_kw' in df.columns, "can't add ITC without solar"
        model.Solar = Param(model.T, initialize=dict(zip(model.T.keys(), df.solar_output_kw.tolist())), doc='solar')

        def itc_constraint(model, t):
            return model.Ein[t] <= model.Solar[t]
        model.itc_constraint = Constraint(model.T, rule=itc_constraint)
        
        

    # Line limit:
    def line_limit(model, t):
        return (model.Output[t] + model.Ein[t] - model.Eout[t]) >= model.Line_Max

    model.line_limit = Constraint(model.T, rule=line_limit)


    # Storage State
    def storage_state(model, t):
        'Storage changes with flows in/out and efficiency losses'
        # Set first hour state of charge to half of max
        if t == model.T.first():
            return model.S[t] == 0
        else:
            return (model.S[t] == (model.S[t - 1]
                                   + dih * (model.Ein[t - 1] * np.sqrt(eff))
                                   - dih * (model.Eout[t - 1] / np.sqrt(eff))))

    model.charge_state = Constraint(model.T, rule=storage_state)

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
        return model.Eout[t] <= model.S[t] * np.sqrt(eff) / dih

    model.positive_charge = Constraint(model.T, rule=positive_charge)

    # TODO make sure  there are no limits on daily discharge limit
    def daily_discharge_limit(model, t):
        """Limit on discharge within a 24 hour period"""
        max_t = model.T.last()
        num_intervals = int(24/dih)
        # Check all t until the last 24 hours
        # No need to check with < 24 hours remaining because the constraint is
        # already in place for a larger number of hours
        if t < max_t - num_intervals + 2:
            return dih * sum(model.Eout[i] for i in range(t, t + num_intervals)) <= model.Dmax
        else:
            return Constraint.Skip

    model.limit_out = Constraint(model.T, rule=daily_discharge_limit)

    # Define the battery income, expenses, and profit
    # Income: (VDER income must be multiplied by dih since the compensation is $/kwh)
    vder_income = sum(model.Vder[t] * (model.Eout[t]) for t in model.T) * dih
    income = vder_income
    model.income = income
    expenses = dih * sum(model.Vder[t] * (model.Ein[t]) for t in model.T)
    model.expenses = expenses
    cost = expenses - income
    model.P = cost
    model.objective = Objective(expr=cost, sense=minimize)

    # Solve the model
    solver = SolverFactory('gurobi')
    solver.solve(model)
    
    
    intervals = range(model.T.first(), model.T.last() + 1)
    Ein = [model.Ein[i].value for i in intervals]
    Eout = [model.Eout[i].value for i in intervals]
    charge_state = [model.S[i].value for i in intervals]
    output = [model.Output[i] for i in intervals]
    if project_type == 'solar+storage':
        solar = [model.Solar[i] for i in intervals]
    else:
        solar = [0 for i in intervals]
    system_out = [(model.Eout[i].value + model.Output[i] - model.Ein[i].value) for i in intervals]
    df_dict = dict(
        intervals=intervals,
        solar=solar,
        output=output,
        Ein=Ein,
        Eout=Eout,
        charge_state=charge_state,
        system_out=system_out,
    )

    df = pd.DataFrame(df_dict)

    return df, model


    
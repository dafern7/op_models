import numpy as np
from pyomo.environ import *
import pandas as pd
import helper
import os

# TODO: add ITC constraint. (Done)
# TODO: add dih (Done)
# TODO: revise filter_damand_rates (Done)
# TODO: revise filter data
# TODO: Change Dmax so it factors in a day consumption instead of a year
# TODO "
#  1) add VDER Constraint (VDER Would be available only when Eout is higher than load + solar) (Done)
#  2) add parameter for building power (Done)
#  3) add parameter for solar (Done)
#  4) add constraint for ITC (Ein is less or equal solar output / 0.8) (Done)
#  5) add revenue values for VDER (LBMP + LSRV) + (ICAP + DRV)
#  6) adding demand to the equation (Done)
#  7) adding the constraints that will keep the demand optimization linear (Done)
#  8) Change time step from hour to 15 minutes OR even better specified time interval. (Done)"


def vps(df, power, capacity, max_discharges, eff=.8, project_type='storage only', itc=False):
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
    # df = df.loc[first_model_hour:last_model_hour, :]
    df = filter_data(df, project_type)

    # dih: data interval hours
    dih = (df.date.iloc[1] - df.date.iloc[0])
    dih = dih.total_seconds() / 3600
    dih = abs(dih)

    # TODO check back this function
    demand_categories, demand_rates_dic = filter_demand_rates(df)

    # Define model sets and parameters
    model = ConcreteModel()
    model.T = Set(doc='index', initialize=(df.index + 1).tolist(), ordered=True)
    model.DemandCategories = Set(initialize=demand_categories, doc='demand rate categories', ordered=True)
    model.Rmax = Param(initialize=power, doc='Max rate of power flow (kW) in or out')
    model.Smax = Param(initialize=capacity, doc='Max storage (kWh)')
    model.Dmax = Param(initialize=capacity * max_discharges, doc='Max discharge in 24 hour period')
    model.Output = Param(model.T, initialize=dict(zip(model.T.keys(), df.output.tolist())),
                         doc='original building power or post solar')

    model.DemandRates = Param(model.DemandCategories, initialize=demand_rates_dic)
    model.Vder = Param(model.T, initialize=dict(zip(list(model.T), df.vder.tolist())), doc='VDER RATES')
    model.Erate = Param(model.T, initialize=dict(zip(list(model.T), df.energy_rate.tolist())), doc='ENERGY RATES')

    # variables: Charge, discharge, binary variable, state of charge, Pmax
    model.Ein = Var(model.T, domain=NonNegativeReals, initialize=0.0)
    model.Eout = Var(model.T, domain=NonNegativeReals, initialize=0.0)
    model.S = Var(model.T, bounds=(0, model.Smax), initialize=model.Smax)
    model.Pmax = Var(model.DemandCategories, domain=NonNegativeReals, initialize=600)
    model.Qplus = Var(model.T, bounds=(0, model.Rmax), initialize=0)
    # model.Qnet = summation(model.Output + model.Ein - model.Eout)

    # set itc constraint
    if itc:
        assert 'solar_output_kw' in df.columns, "can't add ITC without solar"
        model.SolarOut = Param(model.T, initialize=dict(zip(model.T.keys(), df.solar_output_kw.tolist())), doc='solar')

        def itc_constraint(model, t):
            return model.Ein[t] <= model.SolarOut[t] / .8
        model.itc_constraint = Constraint(model.T, rule=itc_constraint)

    #QNet:
    def q_constraint(model, t):
        return model.Output[t] + model.Ein[t] - model.Eout[t] <= model.Qplus[t]

    model.q_constraint = Constraint(model.T, rule=q_constraint)

    # Pmax Constraint
    def power_constraint(model, t):
        rate_cat = df.demand_rate_category.iloc[t - 1]
        return (model.Output[t] + model.Ein[t] - model.Eout[t]) <= model.Pmax[rate_cat]

    model.power_constraint = Constraint(model.T, rule=power_constraint)

    # Storage State
    def storage_state(model, t):
        'Storage changes with flows in/out and efficiency losses'
        # Set first hour state of charge to half of max
        if t == model.T.first():
            return model.S[t] == 0 #model.Smax  # / 2 (remove the hash if you want to start with half power instead)
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

    # Without a constraint the model would discharge in the final hour
    # even when SOC was 0.
    def positive_charge(model, t):
        'Limit discharge to the amount of charge in battery, including losses'
        return model.Eout[t] <= model.S[t]*np.sqrt(eff) / dih
    model.positive_charge = Constraint(model.T, rule=positive_charge)

    def daily_discharge_limit(model, t):
        """Limit on discharge within a 24 hour period"""
        max_t = model.T.last()
        num_intervals = int(24/dih)
        # Check all t until the last 24 hours
        # No need to check with < 24 hours remaining because the constraint is
        # already in place for a larger number of hours
        if t < max_t - num_intervals:
            return dih * sum(model.Eout[i] for i in range(t, t + num_intervals)) <= model.Dmax
        else:
            return Constraint.Skip
    # model.limit_out = Constraint(model.T, rule=daily_discharge_limit)

    # Define the battery income, expenses, and profit
    # Income:
    income = sum(model.Vder[t] * (model.Output[t] + model.Ein[t] - model.Eout[t] - model.Qplus[t]) for t in model.T)
    # Expenses: Energy:
    energy_expenses = 0#dih * sum(model.Erate[t] * model.Qplus[t] for t in model.T)
    # Expenses: Demand
    demand_expenses = sum(model.Pmax[key] * model.DemandRates[key] for key in model.Pmax.keys())
    expenses = energy_expenses + demand_expenses
    cost = expenses + income
    model.P = cost
    model.objective = Objective(expr=cost, sense=minimize)

    # Solve the model
    solver = SolverFactory('gurobi')
    solver.solve(model)

    return model_to_df(model), model


def vps1(df, power, capacity, max_discharges, eff=.8, project_type='storage only', itc=False):
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
    # df = df.loc[first_model_hour:last_model_hour, :]
    df = filter_data(df, project_type)

    # dih: data interval hours
    dih = (df.date.iloc[1] - df.date.iloc[0])
    dih = dih.total_seconds() / 3600
    dih = abs(dih)

    # TODO check back this function
    demand_categories, demand_rates_dic = filter_demand_rates(df)

    # Define model sets and parameters
    model = ConcreteModel()
    model.T = Set(doc='index', initialize=(df.index + 1).tolist(), ordered=True)
    model.DemandCategories = Set(initialize=demand_categories, doc='demand rate categories', ordered=True)
    model.Rmax = Param(initialize=power, doc='Max rate of power flow (kW) in or out')
    model.Smax = Param(initialize=capacity, doc='Max storage (kWh)')
    model.Dmax = Param(initialize=capacity * max_discharges, doc='Max discharge in 24 hour period')
    model.Output = Param(model.T, initialize=dict(zip(model.T.keys(), df.output.tolist())),
                         doc='original building power or post solar')

    model.DemandRates = Param(model.DemandCategories, initialize=demand_rates_dic)
    model.Vder = Param(model.T, initialize=dict(zip(list(model.T), df.vder.tolist())), doc='VDER RATES')
    model.Erate = Param(model.T, initialize=dict(zip(list(model.T), df.energy_rate.tolist())), doc='ENERGY RATES')

    # variables: Charge, discharge, binary variable, state of charge, Pmax
    model.Ein = Var(model.T, domain=NonNegativeReals, initialize=0.0)
    model.Eout = Var(model.T, domain=NonNegativeReals, initialize=0.0)
    model.S = Var(model.T, bounds=(0, model.Smax), initialize=model.Smax)
    model.Pmax = Var(model.DemandCategories, domain=NonNegativeReals, initialize=600)

    # model.Qnet = summation(model.Output + model.Ein - model.Eout)

    # set itc constraint
    if itc:
        assert 'solar_output_kw' in df.columns, "can't add ITC without solar"
        model.SolarOut = Param(model.T, initialize=dict(zip(model.T.keys(), df.solar_output_kw.tolist())), doc='solar')

        def itc_constraint(model, t):
            return model.Ein[t] <= model.SolarOut[t] / .8
        model.itc_constraint = Constraint(model.T, rule=itc_constraint)

    #QNet:
    def q_constraint(model, t):
        return model.Ein[t] + model.Eout[t] <= model.Rmax

    model.q_constraint = Constraint(model.T, rule=q_constraint)

    # Pmax Constraint
    def power_constraint(model, t):
        rate_cat = df.demand_rate_category.iloc[t - 1]
        return (model.Output[t] + model.Ein[t] - model.Eout[t]) <= model.Pmax[rate_cat]

    model.power_constraint = Constraint(model.T, rule=power_constraint)

    # Storage State
    def storage_state(model, t):
        'Storage changes with flows in/out and efficiency losses'
        # Set first hour state of charge to half of max
        if t == model.T.first():
            return model.S[t] == 0 #model.Smax  # / 2 (remove the hash if you want to start with half power instead)
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

    # Without a constraint the model would discharge in the final hour
    # even when SOC was 0.
    def positive_charge(model, t):
        'Limit discharge to the amount of charge in battery, including losses'
        return model.Eout[t] <= model.S[t]*np.sqrt(eff) / dih
    model.positive_charge = Constraint(model.T, rule=positive_charge)

    def daily_discharge_limit(model, t):
        """Limit on discharge within a 24 hour period"""
        max_t = model.T.last()
        num_intervals = int(24/dih)
        # Check all t until the last 24 hours
        # No need to check with < 24 hours remaining because the constraint is
        # already in place for a larger number of hours
        if t < max_t - num_intervals:
            return dih * sum(model.Eout[i] for i in range(t, t + num_intervals)) <= model.Dmax
        else:
            return Constraint.Skip
    # model.limit_out = Constraint(model.T, rule=daily_discharge_limit)

    # Define the battery income, expenses, and profit
    # Income:
    income = sum(model.Vder[t] * model.Eout[t] for t in model.T)
    # Expenses: Energy:
    energy_expenses = 0 #dih * sum(model.Erate[t] * model.Qplus[t] for t in model.T)
    # Expenses: Demand
    demand_expenses = sum(model.Pmax[key] * model.DemandRates[key] for key in model.Pmax.keys())
    expenses = energy_expenses + demand_expenses
    cost = expenses - income
    model.P = cost
    model.objective = Objective(expr=cost, sense=minimize)

    # Solve the model
    solver = SolverFactory('gurobi')
    solver.solve(model)

    return model_to_df(model), model


def vps2(df, power, capacity, max_discharges, eff=.8, project_type='storage only', itc=False):
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
    # df = df.loc[first_model_hour:last_model_hour, :]
    df = filter_data(df, project_type)

    # dih: data interval hours
    dih = (df.date.iloc[1] - df.date.iloc[0])
    dih = dih.total_seconds() / 3600
    dih = abs(dih)

    # TODO check back this function
    demand_categories, demand_rates_dic = filter_demand_rates(df)

    # Define model sets and parameters
    model = ConcreteModel()
    model.T = Set(doc='index', initialize=(df.index + 1).tolist(), ordered=True)
    model.DemandCategories = Set(initialize=demand_categories, doc='demand rate categories', ordered=True)
    model.Rmax = Param(initialize=power, doc='Max rate of power flow (kW) in or out')
    model.Smax = Param(initialize=capacity, doc='Max storage (kWh)')
    model.Dmax = Param(initialize=capacity * max_discharges, doc='Max discharge in 24 hour period')
    model.Output = Param(model.T, initialize=dict(zip(model.T.keys(), df.output.tolist())),
                         doc='original building power or post solar')

    model.DemandRates = Param(model.DemandCategories, initialize=demand_rates_dic)
    model.Vder = Param(model.T, initialize=dict(zip(list(model.T), df.vder.tolist())), doc='VDER RATES')
    model.Erate = Param(model.T, initialize=dict(zip(list(model.T), df.energy_rate.tolist())), doc='ENERGY RATES')

    # variables: Charge, discharge, binary variable, state of charge, Pmax
    model.Ein = Var(model.T, domain=NonNegativeReals, initialize=0.0)
    model.Eout = Var(model.T, domain=NonNegativeReals, initialize=0.0)
    model.X = Var(model.T, domain=Binary, initialize=0.0)
    model.S = Var(model.T, bounds=(0, model.Smax), initialize=model.Smax)
    model.Pmax = Var(model.DemandCategories, domain=NonNegativeReals, initialize=600)

    # model.Qnet = summation(model.Output + model.Ein - model.Eout)

    # set itc constraint
    if itc:
        assert 'solar_output_kw' in df.columns, "can't add ITC without solar"
        model.SolarOut = Param(model.T, initialize=dict(zip(model.T.keys(), df.solar_output_kw.tolist())), doc='solar')

        def itc_constraint(model, t):
            return model.Ein[t] <= model.SolarOut[t] / .8
        model.itc_constraint = Constraint(model.T, rule=itc_constraint)

    # Pmax Constraint
    def power_constraint(model, t):
        rate_cat = df.demand_rate_category.iloc[t - 1]
        return (model.Output[t] + model.Ein[t] - model.Eout[t]) <= model.Pmax[rate_cat]

    model.power_constraint = Constraint(model.T, rule=power_constraint)

    # Storage State
    def storage_state(model, t):
        'Storage changes with flows in/out and efficiency losses'
        # Set first hour state of charge to half of max
        if t == model.T.first():
            return model.S[t] == 0 #model.Smax  # / 2 (remove the hash if you want to start with half power instead)
        else:
            return (model.S[t] == (model.S[t - 1]
                                   + dih * (model.Ein[t-1] * np.sqrt(eff))
                                   - dih * (model.Eout[t-1] / np.sqrt(eff))))

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
        return model.Eout[t] <= model.S[t]*np.sqrt(eff) / dih
    model.positive_charge = Constraint(model.T, rule=positive_charge)

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
    #model.limit_out = Constraint(model.T, rule=daily_discharge_limit)

    # Define the battery income, expenses, and profit
    # Income:
    income = sum(model.Vder[t] * model.Eout[t] for t in model.T)
    # Expenses: Energy:
    energy_expenses = dih * sum(model.Erate[t] * (model.Output[t] + model.Ein[t] - model.Eout[t]) for t in model.T)
    #
    # Expenses: Demand
    demand_expenses = sum(model.Pmax[key] * model.DemandRates[key] for key in model.Pmax.keys())
    expenses = energy_expenses + demand_expenses
    cost = expenses - income
    model.P = cost
    model.objective = Objective(expr=cost, sense=minimize)

    # Solve the model
    solver = SolverFactory('gurobi')
    solver.solve(model)

    return model_to_df(model), model


def lsrv(df, power, capacity, max_discharges, eff=.8, project_type='storage only', itc=False):
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
    # df = df.loc[first_model_hour:last_model_hour, :]
    df = filter_data(df, project_type)

    # dih: data interval hours
    dih = (df.date.iloc[1] - df.date.iloc[0])
    dih = dih.total_seconds() / 3600
    dih = abs(dih)
    dih = 1

    # TODO check back this function
    demand_categories, demand_rates_dic = filter_demand_rates(df)

    # Define model sets and parameters
    model = ConcreteModel()
    model.T = Set(doc='index', initialize=(df.index + 1).tolist(), ordered=True)
    model.DemandCategories = Set(initialize=demand_categories, doc='demand rate categories', ordered=True)
    model.Rmax = Param(initialize=power, doc='Max rate of power flow (kW) in or out')
    model.Smax = Param(initialize=capacity, doc='Max storage (kWh)')
    model.Dmax = Param(initialize=capacity * max_discharges, doc='Max discharge in 24 hour period')
    #model.LSRV_Rate = Param(initialize=float(df.loc[df.lsrv > 0, 'lsrv'].sample()), doc='lsrv value')
    model.Line_Max = Param(initialize=-5000, doc='maximum line power')
    model.Output = Param(model.T, initialize=dict(zip(model.T.keys(), df.output.tolist())),
                         doc='original building power or post solar')
    model.Erate = Param(model.T, initialize=dict(zip(list(model.T), df.energy_rate.tolist())), doc='ENERGY RATES')

    model.DemandRates = Param(model.DemandCategories, initialize=demand_rates_dic)
    model.Vder = Param(model.T, initialize=dict(zip(list(model.T), df.vder.tolist())), doc='VDER RATES')

    # variables: Charge, discharge, binary variable, state of charge, Pmax
    model.Ein = Var(model.T, domain=NonNegativeReals, initialize=0.0)
    model.Eout = Var(model.T, domain=NonNegativeReals, initialize=0.0)
    model.X = Var(model.T, domain=Binary, initialize=0.0)
    model.S = Var(model.T, bounds=(0, model.Smax), initialize=model.Smax)
    model.LSRV = Var(domain=NonNegativeReals, initialize=0)
    model.Pmax = Var(model.DemandCategories, domain=NonNegativeReals, initialize=600)

    # model.Qnet = summation(model.Output + model.Ein - model.Eout)

    # set itc constraint
    if itc:
        assert 'solar_output_kw' in df.columns, "can't add ITC without solar"
        model.SolarOut = Param(model.T, initialize=dict(zip(model.T.keys(), df.solar_output_kw.tolist())), doc='solar')

        def itc_constraint(model, t):
            return model.Ein[t] <= model.SolarOut[t]
        model.itc_constraint = Constraint(model.T, rule=itc_constraint)

    # Pmax Constraint
    def power_constraint(model, t):
        rate_cat = df.demand_rate_category.iloc[t - 1]
        return (-model.Output[t] - model.Ein[t] + model.Eout[t]) >= model.Pmax[rate_cat]

    model.power_constraint = Constraint(model.T, rule=power_constraint)

    # LSRV Constraint
    def lsrv_constraint(model, t):
        if df.lsrv.iloc[t - 1] > 0:
            return (model.Output[t] + model.Ein[t] - model.Eout[t]) >= model.LSRV
        else:
            return Constraint.Skip

    #model.lsrv_constraint = Constraint(model.T, rule=lsrv_constraint)

    # Line limit:
    def line_limit(model, t):
        return (model.Output[t] + model.Ein[t] - model.Eout[t]) >= model.Line_Max

    # model.line_limit = Constraint(model.T, rule=line_limit)

    def no_grid(model, t):
        return (model.Eout[t] - model.Ein[t] - model.Output[t]) >= 0

    # model.no_grid = Constraint(model.T, rule=no_grid)

    # Storage State
    def storage_state(model, t):
        'Storage changes with flows in/out and efficiency losses'
        # Set first hour state of charge to half of max
        if t == model.T.first():
            return model.S[t] == model.Smax  # / 2 (remove the hash if you want to start with half power instead)
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
    # Income:
    vder_income = sum(model.Vder[t] * (model.Eout[t]) for t in model.T)
    lsrv_income = sum(model.Pmax[key] * model.DemandRates[key] for key in model.Pmax.keys())
    income = vder_income + lsrv_income

    expenses = dih * sum(model.Erate[t] * (model.Output[t] + model.Ein[t] - model.Eout[t]) for t in model.T)
    cost = expenses - income
    model.P = cost
    model.objective = Objective(expr=cost, sense=minimize)

    # Solve the model
    solver = SolverFactory('gurobi')
    solver.solve(model)

    return model_to_df(model), model


def peak_shave_coned(df, power, capacity, max_discharges, eff=.8, project_type='storage only', itc=False):
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
    # df = df.loc[first_model_hour:last_model_hour, :]
    df = filter_data(df, project_type)

    # dih: data interval hours
    dih = (df.date.iloc[1] - df.date.iloc[0])
    dih = dih.total_seconds() / 3600
    dih = abs(dih)

    # TODO check back this function
    demand_categories_list, demand_rates_dic_list = filter_demand_rates(df, num_columns=3)
    #print(demand_categories_list)
    #print(demand_rates_dic_list)

    # Define model sets and parameters
    model = ConcreteModel()
    model.T = Set(doc='index', initialize=(df.index + 1).tolist(), ordered=True)
    model.DemandCategories1 = Set(initialize=demand_categories_list[0], doc='demand rate categories1', ordered=True)
    model.DemandCategories2 = Set(initialize=demand_categories_list[1], doc='demand rate categories2', ordered=True)
    model.DemandCategories3 = Set(initialize=demand_categories_list[2], doc='demand rate categories3', ordered=True)

    model.Rmax = Param(initialize=power, doc='Max rate of power flow (kW) in or out')
    model.Smax = Param(initialize=capacity, doc='Max storage (kWh)')
    model.Dmax = Param(initialize=capacity * max_discharges, doc='Max discharge in 24 hour period')

    model.Output = Param(model.T, initialize=dict(zip(model.T.keys(), df.output.tolist())),
                         doc='original building power or post solar')

    model.DemandRates1 = Param(model.DemandCategories1, initialize=demand_rates_dic_list[0])
    model.DemandRates2 = Param(model.DemandCategories2, initialize=demand_rates_dic_list[1])
    model.DemandRates3 = Param(model.DemandCategories3, initialize=demand_rates_dic_list[2])
    model.Vder = Param(model.T, initialize=dict(zip(list(model.T), df.vder.tolist())), doc='VDER RATES')
    model.Erate = Param(model.T, initialize=dict(zip(list(model.T), df.energy_rate.tolist())), doc='ENERGY RATES')

    # variables: Charge, discharge, binary variable, state of charge, Pmax
    model.Ein = Var(model.T, domain=NonNegativeReals, initialize=0.0)
    model.Eout = Var(model.T, domain=NonNegativeReals, initialize=0.0)
    model.Xi = Var(model.T, domain=Binary, initialize=1)
    model.S = Var(model.T, bounds=(0, model.Smax), initialize=model.Smax)
    model.Pmax1 = Var(model.DemandCategories1, domain=NonNegativeReals, initialize=600)
    model.Pmax2 = Var(model.DemandCategories2, domain=NonNegativeReals, initialize=600)
    model.Pmax3 = Var(model.DemandCategories3, domain=NonNegativeReals, initialize=600)

    # set itc constraint
    if itc:
        assert 'solar_output_kw' in df.columns, "can't add ITC without solar"
        model.SolarOut = Param(model.T, initialize=dict(zip(model.T.keys(), df.solar_output_kw.tolist())), doc='solar')

        def itc_constraint(model, t):
            return model.Ein[t] <= model.SolarOut[t] / .8
        model.itc_constraint = Constraint(model.T, rule=itc_constraint)

    # Pmax Constraint
    def power_constraint1(model, t):
        rate_cat = df.demand_rate_category1.iloc[t - 1]
        return (model.Output[t] + model.Ein[t] - model.Eout[t]) <= model.Pmax1[rate_cat]

    model.power_constraint1 = Constraint(model.T, rule=power_constraint1)

    def power_constraint2(model, t):
        rate_cat = df.demand_rate_category2.iloc[t - 1]
        return (model.Output[t] + model.Ein[t] - model.Eout[t]) <= model.Pmax2[rate_cat]

    model.power_constraint2 = Constraint(model.T, rule=power_constraint2)

    def power_constraint3(model, t):
        rate_cat = df.demand_rate_category3.iloc[t - 1]
        return (model.Output[t] + model.Ein[t] - model.Eout[t]) <= model.Pmax3[rate_cat]

    model.power_constraint3 = Constraint(model.T, rule=power_constraint3)

    # Storage State
    def storage_state(model, t):
        'Storage changes with flows in/out and efficiency losses'
        # Set first hour state of charge to half of max
        if t == model.T.first():
            return model.S[t] == 0 # model.Smax
        else:
            return (model.S[t] == (model.S[t - 1]
                                   + dih * (model.Ein[t - 1] * np.sqrt(eff))
                                   - dih * (model.Eout[t - 1] / np.sqrt(eff))))

    model.charge_state = Constraint(model.T, rule=storage_state)

    def discharge_constraint(model, t):
        """Maximum discharge within a single hour"""
        return model.Eout[t] <= model.Rmax * model.Xi[t]

    model.discharge = Constraint(model.T, rule=discharge_constraint)

    def charge_constraint(model, t):
        """Maximum charge within a single hour"""
        return model.Ein[t] <= model.Rmax * (1 - model.Xi[t])

    model.charge = Constraint(model.T, rule=charge_constraint)

    # Without a constraint the model would discharge in the final hour
    # even when SOC was 0.
    def positive_charge(model, t):
        'Limit discharge to the amount of charge in battery, including losses'
        return model.Eout[t] <= model.S[t]*np.sqrt(eff) / dih
    model.positive_charge = Constraint(model.T, rule=positive_charge)

    def daily_discharge_limit(model, t):
        """Limit on discharge within a 24 hour period"""
        max_t = model.T.last()
        num_intervals = int(24/dih)
        # Check all t until the last 24 hours
        # No need to check with < 24 hours remaining because the constraint is
        # already in place for a larger number of hours
        if t < max_t - num_intervals:
            return dih * sum(model.Eout[i] for i in range(t, t + num_intervals)) <= model.Dmax
        else:
            return Constraint.Skip
    # model.limit_out = Constraint(model.T, rule=daily_discharge_limit)

    # Define the battery income, expenses, and profit
    # Expenses: Energy:
    energy_expenses = dih * sum(model.Erate[t] * (model.Output[t] + model.Ein[t] - model.Eout[t]) for t in model.T)
    # Expenses: Demand
    demand_expenses = (sum(model.Pmax1[key] * model.DemandRates1[key] for key in model.Pmax1.keys())
                       + sum(model.Pmax2[key] * model.DemandRates2[key] for key in model.Pmax2.keys())
                       + sum(model.Pmax3[key] * model.DemandRates3[key] for key in model.Pmax3.keys()))
    expenses = energy_expenses + demand_expenses
    model.P = expenses
    model.objective = Objective(expr=expenses, sense=minimize)

    # Solve the model
    solver = SolverFactory('gurobi')
    solver.solve(model)

    return model_to_df(model), model


def optimize_df(df, power, capacity, max_discharges, eff=.8, project_type='storage only', itc=False):
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
    # df = df.loc[first_model_hour:last_model_hour, :]
    df = filter_data(df, project_type)

    # dih: data interval hours
    dih = (df.date.iloc[1] - df.date.iloc[0])
    dih = dih.total_seconds() / 3600
    dih = abs(dih)

    # TODO check back this function
    demand_categories, demand_rates_dic = filter_demand_rates(df)

    # Define model sets and parameters
    model = ConcreteModel()
    model.T = Set(doc='index', initialize=(df.index + 1).tolist(), ordered=True)
    model.DemandCategories = Set(initialize=demand_categories, doc='demand rate categories', ordered=True)
    model.Rmax = Param(initialize=power, doc='Max rate of power flow (kW) in or out')
    model.Smax = Param(initialize=capacity, doc='Max storage (kWh)')
    model.Dmax = Param(initialize=capacity * max_discharges, doc='Max discharge in 24 hour period')
    model.Output = Param(model.T, initialize=dict(zip(model.T.keys(), df.output.tolist())),
                         doc='original building power or post solar')

    model.DemandRates = Param(model.DemandCategories, initialize=demand_rates_dic)
    model.Vder = Param(model.T, initialize=dict(zip(list(model.T), df.vder.tolist())), doc='VDER RATES')
    model.Erate = Param(model.T, initialize=dict(zip(list(model.T), df.energy_rate.tolist())), doc='ENERGY RATES')

    # variables: Charge, discharge, binary variable, state of charge, Pmax
    model.Ein = Var(model.T, domain=NonNegativeReals, initialize=0.0)
    model.Eout = Var(model.T, domain=NonNegativeReals, initialize=0.0)
    model.Xi = Var(model.T, domain=Binary, initialize=1)
    model.S = Var(model.T, bounds=(0, model.Smax), initialize=model.Smax)
    model.Pmax = Var(model.DemandCategories, domain=NonNegativeReals, initialize=600)

    # set itc constraint
    if itc:
        assert 'solar_output_kw' in df.columns, "can't add ITC without solar"
        model.SolarOut = Param(model.T, initialize=dict(zip(model.T.keys(), df.solar_output_kw.tolist())), doc='solar')

        def itc_constraint(model, t):
            return model.Ein[t] <= model.SolarOut[t] / .8
        model.itc_constraint = Constraint(model.T, rule=itc_constraint)

    # Pmax Constraint
    def power_constraint(model, t):
        rate_cat = df.demand_rate_category.iloc[t - 1]
        return (model.Output[t] + model.Ein[t] - model.Eout[t]) * (1 - model.Xi[t]) <= model.Pmax[rate_cat]

    model.power_constraint = Constraint(model.T, rule=power_constraint)

    # Storage State
    def storage_state(model, t):
        'Storage changes with flows in/out and efficiency losses'
        # Set first hour state of charge to half of max
        if t == model.T.first():
            return model.S[t] == 0 #model.Smax  # / 2 (remove the hash if you want to start with half power instead)
        else:
            return (model.S[t] == (model.S[t - 1]
                                   + dih * (model.Ein[t - 1] * np.sqrt(eff))
                                   - dih * (model.Eout[t - 1] / np.sqrt(eff))))

    model.charge_state = Constraint(model.T, rule=storage_state)

    def excess(model, t):
        "Excess power that will go to the grid"
        return (model.Output[t] + model.Ein[t] - model.Eout[t]) * model.Xi[t] <= 0

    model.Excess = Constraint(model.T, rule=excess)

    def consumption(model, t):
        "Consumption of power that will be billed by utility"
        return (model.Eout[t] - model.Output[t] - model.Ein[t]) * (1 - model.Xi[t]) <= 0

    model.Consumption = Constraint(model.T, rule=consumption)

    def discharge_constraint(model, t):
        """Maximum discharge within a single hour"""
        return model.Eout[t] <= model.Rmax

    model.discharge = Constraint(model.T, rule=discharge_constraint)

    def charge_constraint(model, t):
        """Maximum charge within a single hour"""
        return model.Ein[t] <= model.Rmax

    model.charge = Constraint(model.T, rule=charge_constraint)

    # Without a constraint the model would discharge in the final hour
    # even when SOC was 0.
    def positive_charge(model, t):
        'Limit discharge to the amount of charge in battery, including losses'
        return model.Eout[t] <= model.S[t]*np.sqrt(eff) / dih
    model.positive_charge = Constraint(model.T, rule=positive_charge)

    def daily_discharge_limit(model, t):
        """Limit on discharge within a 24 hour period"""
        max_t = model.T.last()
        num_intervals = int(24/dih)
        # Check all t until the last 24 hours
        # No need to check with < 24 hours remaining because the constraint is
        # already in place for a larger number of hours
        if t < max_t - num_intervals:
            return dih * sum(model.Eout[i] for i in range(t, t + num_intervals)) <= model.Dmax
        else:
            return Constraint.Skip
    # model.limit_out = Constraint(model.T, rule=daily_discharge_limit)

    # Define the battery income, expenses, and profit
    # Income:
    income = sum(model.Vder[t] * (model.Eout[t] - model.Ein[t] - model.Output[t]) * model.Xi[t] for t in model.T)
    # Expenses: Energy:
    energy_expenses = dih * sum(
        model.Erate[t] * (model.Output[t] + model.Ein[t] - model.Eout[t]) * (1 - model.Xi[t]) for t in model.T)
    # Expenses: Demand
    demand_expenses = sum(model.Pmax[key] * model.DemandRates[key] for key in model.Pmax.keys())
    expenses = energy_expenses + demand_expenses
    cost = expenses - income
    model.P = cost
    model.objective = Objective(expr=cost, sense=minimize)

    # Solve the model
    solver = SolverFactory('gurobi')
    solver.solve(model)

    return model_to_df(model), model


# Demand Rate finder
def filter_demand_rates(df, num_columns=1):
    if num_columns >1:
        itr1 = ['demand_rate_category' + str(i) for i in range(1, num_columns+1)]
        itr2 = ['demand_rate' + str(i) for i in range(1, num_columns+1)]
        dics = []
        cats = []
        for i1, i2 in zip(itr1, itr2):
            rate_categories = []
            dic = {}
            for rate_cat in set(df[i1]):
                rate_categories.append(rate_cat)
                dic[rate_cat] = df.loc[df[i1] == rate_cat, i2].values[0]
            dics.append(dic)
            cats.append(rate_categories)
        return cats, dics
    else:
        rate_categories = []
        dic = {}
        for rate_cat in set(df.demand_rate_category):
            rate_categories.append(rate_cat)
            dic[rate_cat] = df.loc[df.demand_rate_category == rate_cat, 'demand_rate'].values[0]
        return rate_categories, dic


# Filter data depending on project type and add vder values based on derv,icap3, and lbmpm
def filter_data(df, project_type):
    if project_type == 'solar+storage':
        df['output'] = df['original_building_power_kw'] - df['solar_output_kw']
    elif project_type == 'storage only':
        df['output'] = df['original_building_power_kw']
    else:
        raise AssertionError('Project type is not appropriately defined')

    if 'vder' not in df.columns:
        directory = os.path.dirname(os.path.abspath(__file__))
        directory = os.path.join(directory, 'datasets', 'icap3.csv')
        vder = pd.read_csv(directory)
        df = helper.add_vder_to_df(df, vder)
        df.reset_index(drop=True, inplace=True)

    return df


def model_to_df(model):
    """
    Create a dataframe with hourly charge, discharge, state of charge, and
    price columns from a pyomo model. Only uses data from between the first
    (inclusive) and last (exclusive) hours.

    Parameters
    ----------
    model : pyomo model
        Model that has been solved

    Returns
    -------
    dataframe

    """
    # Need to increase the first & last data point by 1 because of pyomo indexing
    # and add 1 to the value of last model hour because of the range
    intervals = range(model.T.first(), model.T.last() + 1)
    Ein = [model.Ein[i].value for i in intervals]
    Eout = [model.Eout[i].value for i in intervals]
    Qplus =0# [model.Qplus[i].value for i in intervals]
    charge_state = [model.S[i].value for i in intervals]
    output = [model.Output[i] for i in intervals]
    df_dict = dict(
        intervals=intervals,
        Ein=Ein,
        Eout=Eout,
        Qplus=Qplus,
        charge_state=charge_state,
        output=output,
    )

    df = pd.DataFrame(df_dict)

    return df


def optimize_mip(df, power, capacity, max_discharges, eff=.8, project_type='storage only', itc=False,
                 resolution=5, max_dod=.85):
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
    # df = df.loc[first_model_hour:last_model_hour, :]
    df = filter_data(df, project_type)

    # dih: data interval hours
    dih = (df.date.iloc[1] - df.date.iloc[0])
    dih = dih.total_seconds() / 3600
    dih = abs(dih)
    print("dih:",dih)

    # TODO check back this function
    demand_categories, demand_rates_dic = filter_demand_rates(df)

    # Define model sets and parameters
    k = list(range(1, power // resolution + 2))
    e_in = [(idx - 1) * resolution for idx in k]

    model = ConcreteModel()
    model.T = Set(doc='index', initialize=(df.index + 1).tolist(), ordered=True)
    model.K = Set(doc='K', initialize=k, ordered=True)
    model.I = Set(doc='I', initialize=k, ordered=True)
    model.DemandCategories = Set(initialize=demand_categories, doc='demand rate categories', ordered=True)

    model.Smax = Param(initialize=capacity, doc='Max storage (kWh)')
    model.Dmax = Param(initialize=capacity * max_discharges, doc='Max discharge in 24 hour period')
    model.Output = Param(model.T, initialize=dict(zip(model.T.keys(), df.output.tolist())),
                         doc='original building power or post solar')
    model.Ein = Param(model.T, model.K, initialize=get_dic(model, e_in))
    model.Eout = Param(model.T, model.K, initialize=get_dic(model, e_in))

    model.DemandRates = Param(model.DemandCategories, initialize=demand_rates_dic)
    model.Vder = Param(model.T, initialize=dict(zip(list(model.T), df.vder.tolist())), doc='VDER RATES')
    model.Erate = Param(model.T, initialize=dict(zip(list(model.T), df.energy_rate.tolist())), doc='ENERGY RATES')

    # variables: Charge, discharge, binary variable, state of charge, Pmax
    model.Xk = Var(model.T, model.K, domain=Binary, initialize=0)
    model.Xi = Var(model.T, model.I,domain=Binary, initialize=0)
    model.S = Var(model.T, bounds=(0, model.Smax), initialize=model.Smax)
    model.Pmax = Var(model.DemandCategories, domain=NonNegativeReals, initialize=600)

    # set itc constraint
    # TODO change to add itc for the new model
    if itc:
        assert 'solar_output_kw' in df.columns, "can't add ITC without solar"
        model.SolarOut = Param(model.T, initialize=dict(zip(model.T.keys(), df.solar_output_kw.tolist())), doc='solar')

        def itc_constraint(model, t):
            return model.Ein[t] <= model.SolarOut[t] / .8
        model.itc_constraint = Constraint(model.T, rule=itc_constraint)

    # Pmax Constraint
    def power_constraint(model, t):
        rate_cat = df.demand_rate_category.iloc[t - 1]
        out = model.Output[t] * sum(model.Xi[t, i] for i in model.I.keys())
        charge = sum(model.Ein[t, i] * model.Xi[t, i] for i in model.I.keys())
        discharge = sum(model.Eout[t, i] * model.Xi[t, i] for i in model.I.keys())
        return (out + charge - discharge) <= model.Pmax[rate_cat]

    model.power_constraint = Constraint(model.T, rule=power_constraint)

    # Binary Constraint:
    def binary_constraint(model, t):
        return sum(model.Xi[t, k_idx]+model.Xk[t,k_idx] for k_idx in model.K.keys()) == 1

    model.binary_constraint = Constraint(model.T, rule=binary_constraint)

    # Storage State
    def storage_state(model, t):
        'Storage changes with flows in/out and efficiency losses'
        # Set first hour state of charge to half of max
        if t == model.T.first():
            return model.S[t] == model.Smax / 2
        else:
            charged = sum(model.Ein[t-1, k] * (model.Xk[t-1,k] + model.Xi[t-1,k]) for k in model.K.keys())
            discharged = sum(model.Eout[t - 1, k] * (model.Xk[t - 1, k] + model.Xi[t - 1, k]) for k in model.K.keys())
            return (model.S[t] == (model.S[t - 1]
                                   + dih * (charged * np.sqrt(eff))
                                   - dih * (discharged / np.sqrt(eff))))

    model.charge_state = Constraint(model.T, rule=storage_state)

    def excess(model, t):
        "Excess power that will go to the grid"
        out = model.Output[t] * sum(model.Xk[t, k] for k in model.K.keys())
        charge = sum(model.Ein[t, k] * model.Xk[t, k] for k in model.K.keys())
        discharge = sum(model.Eout[t, k] * model.Xk[t, k] for k in model.K.keys())
        return out + charge - discharge <= 0

    model.Excess = Constraint(model.T, rule=excess)

    def consumption(model, t):
        "Consumption of power that will be billed by utility"
        out = model.Output[t] * sum(model.Xi[t, i] for i in model.I.keys())
        charge = sum(model.Ein[t, i] * model.Xi[t, i] for i in model.I.keys())
        discharge = sum(model.Eout[t, i] * model.Xi[t, i] for i in model.I.keys())
        return -(out + charge - discharge) <= 0

    model.Consumption = Constraint(model.T, rule=consumption)

    #
    ##################################################################################################################
    # Without a constraint the model would discharge in the final hour even when SOC was 0.
    def positive_charge(model, t):
        # TODO work this too to be compatible with the new model
        'Limit discharge to the amount of charge in battery, including losses'
        return model.Eout[t] <= model.S[t]*np.sqrt(eff) / dih
    # model.positive_charge = Constraint(model.T, rule=positive_charge)

    def daily_discharge_limit(model, t):
        # TODO Chage this to be compatible with the new model
        """Limit on discharge within a 24 hour period"""
        max_t = model.T.last()
        num_intervals = int(24/dih)
        # Check all t until the last 24 hours
        # No need to check with < 24 hours remaining because the constraint is
        # already in place for a larger number of hours
        if t < max_t - num_intervals:
            return dih * sum(model.Eout[i] for i in range(t, t + num_intervals)) <= model.Dmax
        else:
            return Constraint.Skip
    # model.limit_out = Constraint(model.T, rule=daily_discharge_limit)
    #################################################################################################################

    # Define the battery income, expenses, and profit
    # Income:
    vder_credits = 0  # vder credits are negative values here (or zero)
    energy_expenses = 0
    for t in model.T.keys():
        out = model.Output[t] * sum(model.Xk[t, k] for k in model.K.keys())
        charge = sum(model.Ein[t, k] * model.Xk[t, k] for k in model.K.keys())
        discharge = sum(model.Eout[t, k] * model.Xk[t, k] for k in model.K.keys())
        vder_credits += (out + charge - discharge) * model.Vder[t]

        out = model.Output[t] * sum(model.Xi[t, i] for i in model.I.keys())
        charge = sum(model.Ein[t, i] * model.Xi[t, i] for i in model.I.keys())
        discharge = sum(model.Eout[t, i] * model.Xi[t, i] for i in model.I.keys())
        energy_expenses += dih * (out + charge - discharge) * model.Erate[t]

    demand_expenses = sum(model.Pmax[key] * model.DemandRates[key] for key in model.Pmax.keys())

    model.EnergyExpenses = energy_expenses
    model.DemandEx = demand_expenses
    cost = vder_credits + energy_expenses + demand_expenses
    model.P = cost
    model.objective = Objective(expr=cost, sense=minimize)

    # Solve the model
    #executable = r'C:\Program Files\IBM\ILOG\CPLEX_Studio129\cplex\bin\x64_win64\cplex'
    #solver = SolverFactory('cplex', executable=executable)
    solver = SolverFactory('gurobi')
    solver.solve(model)

    return mip_model_to_df(model), model


def get_dic(model, e_in):
    dic = {}
    for i in model.T.keys():
        for k in model.K.keys():
            dic[(i, k)] = e_in[k - 1]

    return dic


def mip_model_to_df(model):
    """
    Create a dataframe with hourly charge, discharge, state of charge, and
    price columns from a pyomo model. Only uses data from between the first
    (inclusive) and last (exclusive) hours.

    Parameters
    ----------
    model : pyomo model
        Model that has been solved

    Returns
    -------
    dataframe

    """
    # Need to increase the first & last data point by 1 because of pyomo indexing
    # and add 1 to the value of last model hour because of the range
    intervals = range(model.T.first(), model.T.last() + 1)
    Ein = [sum(model.Ein[t, k] * (model.Xi[t,k].value+model.Xk[t,k].value) for k in model.K.keys()) for t in intervals]
    Eout = [sum(model.Eout[t,k] * (model.Xi[t,k].value+model.Xk[t,k].value) for k in model.K.keys()) for t in intervals]
    charge_state = [model.S[i].value for i in intervals]
    Xi = [sum(model.Xi[t, k].value for k in model.K.keys()) for t in intervals]
    Xk = [sum(model.Xk[t, k].value for k in model.K.keys()) for t in intervals]

    output = [model.Output[t] for t in intervals]
    df_dict = dict(
        intervals=intervals,
        Ein= Ein,
        Eout=Eout,
        charge_state=charge_state,
        Xi=Xi,
        Xk=Xk,
        output=output,
    )

    df = pd.DataFrame(df_dict)

    return df



def optimize_milp(df, power, capacity, max_discharges, max_dod=1,eff=.8, project_type='storage only', itc=False):
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
    # df = df.loc[first_model_hour:last_model_hour, :]
    df = filter_data(df, project_type)

    # dih: data interval hours
    dih = (df.date.iloc[1] - df.date.iloc[0])
    dih = dih.total_seconds() / 3600
    dih = abs(dih)

    # TODO check back this function
    demand_categories, demand_rates_dic = filter_demand_rates(df)
    capacity = capacity * max_dod
    # Define model sets and parameters
    model = ConcreteModel()
    model.T = Set(doc='index', initialize=(df.index + 1).tolist(), ordered=True)
    model.DemandCategories = Set(initialize=demand_categories, doc='demand rate categories', ordered=True)
    model.Rmax = Param(initialize=power, doc='Max rate of power flow (kW) in or out')
    model.Smax = Param(initialize=capacity, doc='Max storage (kWh)')
    model.Dmax = Param(initialize=capacity * max_discharges, doc='Max discharge in 24 hour period')
    model.Output = Param(model.T, initialize=dict(zip(model.T.keys(), df.output.tolist())),
                         doc='original building power or post solar')

    model.DemandRates = Param(model.DemandCategories, initialize=demand_rates_dic)
    model.Vder = Param(model.T, initialize=dict(zip(list(model.T), df.vder.tolist())), doc='VDER RATES')
    model.Erate = Param(model.T, initialize=dict(zip(list(model.T), df.energy_rate.tolist())), doc='ENERGY RATES')

    # variables: Charge, discharge, binary variable, state of charge, Pmax
    model.Ein_x = Var(model.T, domain=NonNegativeReals, initialize=0.0)
    model.Eout_x = Var(model.T, domain=NonNegativeReals, initialize=0.0)
    model.Ein_y = Var(model.T, domain=NonNegativeReals, initialize=0.0)
    model.Eout_y = Var(model.T, domain=NonNegativeReals, initialize=0.0)
    model.X1 = Var(model.T, domain=Binary, initialize=1)
    model.X2 = Var(model.T, domain=Binary, initialize=0)
    model.S = Var(model.T, bounds=(0, model.Smax), initialize=model.Smax)
    model.Pmax = Var(model.DemandCategories, domain=NonNegativeReals, initialize=600)

    # set itc constraint
    if itc:
        assert 'solar_output_kw' in df.columns, "can't add ITC without solar"
        model.SolarOut = Param(model.T, initialize=dict(zip(model.T.keys(), df.solar_output_kw.tolist())), doc='solar')

        def itc_constraint(model, t):
            return model.Ein_x[t] + model.Ein_y[t] <= model.SolarOut[t] / .8
        model.itc_constraint = Constraint(model.T, rule=itc_constraint)

    def binary_constraint(model, t):
        return model.X1[t] + model.X2[t] == 1
    model.binary_constraint = Constraint(model.T, rule=binary_constraint)

    # Pmax Constraint
    def power_constraint(model, t):
        rate_cat = df.demand_rate_category.iloc[t - 1]
        return (model.Output[t] + model.Ein_x[t] + model.Ein_y[t]
                - model.Eout_x[t] - model.Eout_y[t]) <= model.Pmax[rate_cat]

    model.power_constraint = Constraint(model.T, rule=power_constraint)

    # Storage State
    def storage_state(model, t):
        'Storage changes with flows in/out and efficiency losses'
        # Set first hour state of charge to half of max
        if t == model.T.first():
            return model.S[t] == model.Smax  # / 2 (remove the hash if you want to start with half power instead)
        else:
            return (model.S[t] == (model.S[t - 1]
                                   + dih * (model.Ein_x[t - 1] + model.Ein_y[t-1]) * np.sqrt(eff)
                                   - dih * (model.Eout_x[t - 1] + model.Eout_y[t-1]) / np.sqrt(eff)))
    model.charge_state = Constraint(model.T, rule=storage_state)

    def excess(model, t):
        "Excess power that will go to the grid"
        return (model.Output[t] * model.X2[t] + model.Ein_y[t] - model.Eout_y[t]) <= 0
    model.Excess = Constraint(model.T, rule=excess)

    def consumption(model, t):
        "Consumption of power that will be billed by utility"
        return (model.Eout_x[t] - model.Output[t] * model.X1[t] - model.Ein_x[t]) <= 0

    model.Consumption = Constraint(model.T, rule=consumption)

    def discharge_constraint_x(model, t):
        """Maximum discharge within a single hour"""
        return model.Eout_x[t] <= model.Rmax * model.X1[t]
    model.discharge_x = Constraint(model.T, rule=discharge_constraint_x)

    def discharge_constraint_y(model, t):
        """Maximum discharge within a single hour"""
        return model.Eout_y[t] <= model.Rmax * model.X2[t]
    model.discharge_y = Constraint(model.T, rule=discharge_constraint_y)

    def charge_constraint_x(model, t):
        """Maximum charge within a single hour"""
        return model.Ein_x[t] <= model.Rmax * model.X1[t]
    model.charge_x = Constraint(model.T, rule=charge_constraint_x)

    def charge_constraint_y(model, t):
        """Maximum charge within a single hour"""
        return model.Ein_y[t] <= model.Rmax * model.X2[t]
    model.charge_y = Constraint(model.T, rule=charge_constraint_y)

    # TODO depreciated (need adjustments)
    # Without a constraint the model would discharge in the final hour
    # even when SOC was 0.
    def positive_charge(model, t):
        'Limit discharge to the amount of charge in battery, including losses'
        return model.Eout[t] <= model.S[t]*np.sqrt(eff) / dih
    #model.positive_charge = Constraint(model.T, rule=positive_charge)
    # TODO depreciated (need adjustments)
    def daily_discharge_limit(model, t):
        """Limit on discharge within a 24 hour period"""
        max_t = model.T.last()
        num_intervals = int(24/dih)
        # Check all t until the last 24 hours
        # No need to check with < 24 hours remaining because the constraint is
        # already in place for a larger number of hours
        #if t < max_t - num_intervals:
        #    return dih * sum(model.Eout[i] for i in range(t, t + num_intervals)) <= model.Dmax
        #else:
        #    return Constraint.Skip
    # model.limit_out = Constraint(model.T, rule=daily_discharge_limit)

    # Define the battery income, expenses, and profit
    # Income:
    income = sum(model.Vder[t] * (model.Eout_y[t] - model.Ein_y[t] - model.Output[t] * model.X2[t]) for t in model.T)
    # Expenses: Energy:
    energy_expenses = dih * sum(
        model.Erate[t] * (model.Output[t] * model.X1[t] + model.Ein_x[t] - model.Eout_x[t]) for t in model.T)
    # Expenses: Demand
    demand_expenses = sum(model.Pmax[key] * model.DemandRates[key] for key in model.Pmax.keys())
    expenses = energy_expenses + demand_expenses
    cost = expenses - income
    model.P = cost
    model.objective = Objective(expr=cost, sense=minimize)

    # Solve the model
    solver = SolverFactory('gurobi')
    solver.solve(model)

    return milp_model_to_df(model), model


def milp_model_to_df(model):
    """
    Create a dataframe with hourly charge, discharge, state of charge, and
    price columns from a pyomo model. Only uses data from between the first
    (inclusive) and last (exclusive) hours.

    Parameters
    ----------
    model : pyomo model
        Model that has been solved

    Returns
    -------
    dataframe

    """
    # Need to increase the first & last data point by 1 because of pyomo indexing
    # and add 1 to the value of last model hour because of the range
    intervals = range(model.T.first(), model.T.last() + 1)
    Ein_x = [model.Ein_x[t].value for t in intervals]
    Eout_x = [model.Eout_x[t].value for t in intervals]
    Ein_y = [model.Ein_y[t].value for t in intervals]
    Eout_y = [model.Eout_y[t].value for t in intervals]
    charge_state = [model.S[t].value for t in intervals]
    X1 = [model.X1[t].value for t in intervals]
    X2 = [model.X2[t].value for t in intervals]
    output = [model.Output[i] for i in intervals]
    df_dict = dict(
        intervals=intervals,
        Ein_x=Ein_x,
        Ein_y=Ein_y,
        Eout_x=Eout_x,
        Eout_y=Eout_y,
        charge_state=charge_state,
        X1=X1,
        X2=X2,
        output=output,
    )

    df = pd.DataFrame(df_dict)

    return df


def optimize_milp_test(df, power, capacity, max_discharges, max_dod=1,eff=.8, project_type='storage only', itc=False):
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
    # df = df.loc[first_model_hour:last_model_hour, :]
    df = filter_data(df, project_type)

    # dih: data interval hours
    dih = (df.date.iloc[1] - df.date.iloc[0])
    dih = dih.total_seconds() / 3600
    dih = abs(dih)

    # TODO check back this function
    demand_categories, demand_rates_dic = filter_demand_rates(df)
    capacity = capacity * max_dod
    M1 = df.output.max() * 100
    M2 = df.output.max() * 100
    # Define model sets and parameters
    model = ConcreteModel()
    model.T = Set(doc='index', initialize=(df.index + 1).tolist(), ordered=True)
    model.DemandCategories = Set(initialize=demand_categories, doc='demand rate categories', ordered=True)
    model.Rmax = Param(initialize=power, doc='Max rate of power flow (kW) in or out')
    model.Smax = Param(initialize=capacity, doc='Max storage (kWh)')
    model.Dmax = Param(initialize=capacity * max_discharges, doc='Max discharge in 24 hour period')
    model.Output = Param(model.T, initialize=dict(zip(model.T.keys(), df.output.tolist())),
                         doc='original building power or post solar')

    model.DemandRates = Param(model.DemandCategories, initialize=demand_rates_dic)
    model.Vder = Param(model.T, initialize=dict(zip(list(model.T), df.vder.tolist())), doc='VDER RATES')
    model.Erate = Param(model.T, initialize=dict(zip(list(model.T), df.energy_rate.tolist())), doc='ENERGY RATES')

    # variables: Charge, discharge, binary variable, state of charge, Pmax
    model.Ein_x = Var(model.T, domain=NonNegativeReals, initialize=0.0)
    model.Eout_x = Var(model.T, domain=NonNegativeReals, initialize=0.0)
    model.Ein_y = Var(model.T, domain=NonNegativeReals, initialize=0.0)
    model.Eout_y = Var(model.T, domain=NonNegativeReals, initialize=0.0)
    model.X1 = Var(model.T, domain=Binary, initialize=1)
    model.X2 = Var(model.T, domain=Binary, initialize=0)
    model.S = Var(model.T, bounds=(0, model.Smax), initialize=model.Smax)
    model.Pmax = Var(model.DemandCategories, domain=NonNegativeReals, initialize=600)

    # set itc constraint
    if itc:
        assert 'solar_output_kw' in df.columns, "can't add ITC without solar"
        model.SolarOut = Param(model.T, initialize=dict(zip(model.T.keys(), df.solar_output_kw.tolist())), doc='solar')

        def itc_constraint(model, t):
            return model.Ein_x[t] + model.Ein_y[t] <= model.SolarOut[t] / .8
        model.itc_constraint = Constraint(model.T, rule=itc_constraint)

    def binary_constraint(model, t):
        return model.X1[t] + model.X2[t] == 1
    model.binary_constraint = Constraint(model.T, rule=binary_constraint)

    # Pmax Constraint
    def power_constraint(model, t):
        rate_cat = df.demand_rate_category.iloc[t - 1]
        return (model.Output[t] + model.Ein_x[t] + model.Ein_y[t]
                - model.Eout_x[t] - model.Eout_y[t]) <= model.Pmax[rate_cat]

    model.power_constraint = Constraint(model.T, rule=power_constraint)

    # Storage State
    def storage_state(model, t):
        'Storage changes with flows in/out and efficiency losses'
        # Set first hour state of charge to half of max
        if t == model.T.first():
            return model.S[t] == model.Smax  # / 2 (remove the hash if you want to start with half power instead)
        else:
            return (model.S[t] == (model.S[t - 1]
                                   + dih * (model.Ein_x[t - 1] + model.Ein_y[t-1]) * np.sqrt(eff)
                                   - dih * (model.Eout_x[t - 1] + model.Eout_y[t-1]) / np.sqrt(eff)))
    model.charge_state = Constraint(model.T, rule=storage_state)

    def excess(model, t):
        "Excess power that will go to the grid"
        return (model.Output[t] * model.X2[t] + model.Ein_y[t] - model.Eout_y[t]) <= 0
    model.Excess = Constraint(model.T, rule=excess)

    def consumption(model, t):
        "Consumption of power that will be billed by utility"
        return (model.Eout_x[t] - model.Output[t] * model.X1[t] - model.Ein_x[t]) <= 0

    model.Consumption = Constraint(model.T, rule=consumption)

    def discharge_constraint_x(model, t):
        """Maximum discharge within a single hour"""
        return model.Eout_x[t] <= model.Rmax * model.X1[t]
    model.discharge_x = Constraint(model.T, rule=discharge_constraint_x)

    def discharge_constraint_y(model, t):
        """Maximum discharge within a single hour"""
        return model.Eout_y[t] <= model.Rmax * model.X2[t]
    model.discharge_y = Constraint(model.T, rule=discharge_constraint_y)

    def charge_constraint_x(model, t):
        """Maximum charge within a single hour"""
        return model.Ein_x[t] <= model.Rmax * model.X1[t]
    model.charge_x = Constraint(model.T, rule=charge_constraint_x)

    def charge_constraint_y(model, t):
        """Maximum charge within a single hour"""
        return model.Ein_y[t] <= model.Rmax * model.X2[t]
    model.charge_y = Constraint(model.T, rule=charge_constraint_y)

    # TODO depreciated (need adjustments)
    # Without a constraint the model would discharge in the final hour
    # even when SOC was 0.
    def positive_charge(model, t):
        'Limit discharge to the amount of charge in battery, including losses'
        return model.Eout[t] <= model.S[t]*np.sqrt(eff) / dih
    #model.positive_charge = Constraint(model.T, rule=positive_charge)
    # TODO depreciated (need adjustments)
    def daily_discharge_limit(model, t):
        """Limit on discharge within a 24 hour period"""
        max_t = model.T.last()
        num_intervals = int(24/dih)
        # Check all t until the last 24 hours
        # No need to check with < 24 hours remaining because the constraint is
        # already in place for a larger number of hours
        #if t < max_t - num_intervals:
        #    return dih * sum(model.Eout[i] for i in range(t, t + num_intervals)) <= model.Dmax
        #else:
        #    return Constraint.Skip
    # model.limit_out = Constraint(model.T, rule=daily_discharge_limit)

    # Define the battery income, expenses, and profit
    # Income:
    income = sum(model.Vder[t] * (model.Eout_y[t] - model.Ein_y[t] - model.Output[t] * model.X2[t]) for t in model.T)
    # Expenses: Energy:
    energy_expenses = dih * sum(
        model.Erate[t] * (model.Output[t] * model.X1[t] + model.Ein_x[t] - model.Eout_x[t]) for t in model.T)
    # Expenses: Demand
    demand_expenses = sum(model.Pmax[key] * model.DemandRates[key] for key in model.Pmax.keys())
    expenses = energy_expenses + demand_expenses
    cost = expenses - income
    model.P = cost
    model.objective = Objective(expr=cost, sense=minimize)

    # Solve the model
    solver = SolverFactory('gurobi')
    solver.solve(model)

    return milp_model_to_df(model), model


def optimize_df_linearized(df, power, capacity, max_discharges, eff=.8, project_type='storage only', itc=False):
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
    # df = df.loc[first_model_hour:last_model_hour, :]
    df = filter_data(df, project_type)

    # dih: data interval hours
    dih = (df.date.iloc[1] - df.date.iloc[0])
    dih = dih.total_seconds() / 3600
    dih = abs(dih)

    # TODO check back this function
    demand_categories, demand_rates_dic = filter_demand_rates(df)

    # Define model sets and parameters
    model = ConcreteModel()
    model.T = Set(doc='index', initialize=(df.index + 1).tolist(), ordered=True)
    model.DemandCategories = Set(initialize=demand_categories, doc='demand rate categories', ordered=True)
    model.Rmax = Param(initialize=power, doc='Max rate of power flow (kW) in or out')
    model.Smax = Param(initialize=capacity, doc='Max storage (kWh)')
    model.Dmax = Param(initialize=capacity * max_discharges, doc='Max discharge in 24 hour period')
    model.Output = Param(model.T, initialize=dict(zip(model.T.keys(), df.output.tolist())),
                         doc='original building power or post solar')

    model.DemandRates = Param(model.DemandCategories, initialize=demand_rates_dic)
    model.Vder = Param(model.T, initialize=dict(zip(list(model.T), df.vder.tolist())), doc='VDER RATES')
    model.Erate = Param(model.T, initialize=dict(zip(list(model.T), df.energy_rate.tolist())), doc='ENERGY RATES')

    # variables: Charge, discharge, binary variable, state of charge, Pmax
    model.Ein = Var(model.T, bounds=(0,model.Rmax), initialize=0.0)
    model.Eout = Var(model.T, bounds=(0,model.Rmax), initialize=0.0)
    model.Zin = Var(model.T, bounds=(0,model.Rmax), initialize=0.0)
    model.Zout = Var(model.T, bounds=(0,model.Rmax), initialize=0.0)
    model.Xi = Var(model.T, domain=Binary, initialize=1)
    model.S = Var(model.T, bounds=(0, model.Smax), initialize=model.Smax)
    model.Pmax = Var(model.DemandCategories, domain=NonNegativeReals, initialize=600)

    # set itc constraint
    if itc:
        assert 'solar_output_kw' in df.columns, "can't add ITC without solar"
        model.SolarOut = Param(model.T, initialize=dict(zip(model.T.keys(), df.solar_output_kw.tolist())), doc='solar')

        def itc_constraint(model, t):
            return model.Ein[t] <= model.SolarOut[t] / .8
        model.itc_constraint = Constraint(model.T, rule=itc_constraint)

    # Pmax Constraint
    def power_constraint(model, t):
        rate_cat = df.demand_rate_category.iloc[t - 1]
        return (model.Output[t] + model.Ein[t] - model.Eout[t]) <= model.Pmax[rate_cat]

    model.power_constraint = Constraint(model.T, rule=power_constraint)

    # Storage State
    def storage_state(model, t):
        'Storage changes with flows in/out and efficiency losses'
        # Set first hour state of charge to half of max
        if t == model.T.first():
            return model.S[t] == model.Smax  # / 2 (remove the hash if you want to start with half power instead)
        else:
            return (model.S[t] == (model.S[t - 1]
                                   + dih * (model.Ein[t - 1] * np.sqrt(eff))
                                   - dih * (model.Eout[t - 1] / np.sqrt(eff))))
    model.charge_state = Constraint(model.T, rule=storage_state)

    def excess(model, t):
        "Excess power that will go to the grid"
        return (model.Output[t] * model.Xi[t] + model.Zin[t] - model.Zout[t]) <= 0
    model.Excess = Constraint(model.T, rule=excess)

    def consumption(model, t):
        "Consumption of power that will be billed by utility"
        return (model.Eout[t] - model.Output[t] - model.Ein[t]) - \
               (model.Zout[t] - model.Output[t] * model.Xi[t] - model.Zin[t]) <= 0
    model.Consumption = Constraint(model.T, rule=consumption)

    def zin_lim1(model, t):
        return model.Zin[t] <= model.Ein[t]
    model.zin_lim1 = Constraint(model.T, rule=zin_lim1)

    def zin_lim2(model, t):
        return model.Zin[t] >= model.Ein[t] - model.Rmax * (1-model.Xi[t])

    model.zin_lim2 = Constraint(model.T, rule=zin_lim2)

    def zin_lim3(model, t):
        return model.Zin[t] <= model.Rmax * model.Xi[t]
    model.zin_lim3 = Constraint(model.T, rule=zin_lim3)

    def zout_lim1(model, t):
        return model.Zout[t] <= model.Eout[t]
    model.zout_lim1 = Constraint(model.T, rule=zout_lim1)

    def zout_lim2(model, t):
        return model.Zout[t] >= model.Eout[t] - model.Rmax * (1-model.Xi[t])
    model.zout_lim2 = Constraint(model.T, rule=zout_lim2)

    def zout_lim3(model, t):
        return model.Zout[t] <= model.Rmax * model.Xi[t]
    model.zout_lim3 = Constraint(model.T, rule=zout_lim3)

    # Without a constraint the model would discharge in the final hour
    # even when SOC was 0.
    def positive_charge(model, t):
        'Limit discharge to the amount of charge in battery, including losses'
        return model.Eout[t] <= model.S[t]*np.sqrt(eff) / dih
    model.positive_charge = Constraint(model.T, rule=positive_charge)

    def daily_discharge_limit(model, t):
        """Limit on discharge within a 24 hour period"""
        max_t = model.T.last()
        num_intervals = int(24/dih)
        # Check all t until the last 24 hours
        # No need to check with < 24 hours remaining because the constraint is
        # already in place for a larger number of hours
        if t < max_t - num_intervals:
            return dih * sum(model.Eout[i] for i in range(t, t + num_intervals)) <= model.Dmax
        else:
            return Constraint.Skip
    # model.limit_out = Constraint(model.T, rule=daily_discharge_limit)

    # Define the battery income, expenses, and profit
    # Income:
    income = sum(model.Vder[t] * (model.Zout[t] - model.Zin[t] - model.Output[t] * model.Xi[t]) for t in model.T)
    # Expenses: Energy:
    energy_expenses = dih * sum(
        model.Erate[t] * (model.Output[t] + model.Ein[t] - model.Eout[t] -
                          (model.Output[t] * model.Xi[t] + model.Zin[t] - model.Zout[t])) for t in model.T)
    # Expenses: Demand
    demand_expenses = sum(model.Pmax[key] * model.DemandRates[key] for key in model.Pmax.keys())
    expenses = energy_expenses + demand_expenses
    cost = expenses - income
    model.P = cost
    model.objective = Objective(expr=cost, sense=minimize)

    # Solve the model
    solver = SolverFactory('gurobi')
    solver.solve(model)

    return model_to_df(model), model


def vder_peak_shave(df, power, capacity, max_discharges, eff=.8, project_type='storage only', itc=False):
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
    # df = df.loc[first_model_hour:last_model_hour, :]
    df = filter_data(df, project_type)

    # dih: data interval hours
    dih = (df.date.iloc[1] - df.date.iloc[0])
    dih = dih.total_seconds() / 3600
    dih = abs(dih)

    # TODO check back this function
    demand_categories, demand_rates_dic = filter_demand_rates(df)

    # Define model sets and parameters
    model = ConcreteModel()
    model.T = Set(doc='index', initialize=(df.index + 1).tolist(), ordered=True)
    model.DemandCategories = Set(initialize=demand_categories, doc='demand rate categories', ordered=True)
    model.Rmax = Param(initialize=power, doc='Max rate of power flow (kW) in or out')
    model.Smax = Param(initialize=capacity, doc='Max storage (kWh)')
    model.Dmax = Param(initialize=capacity * max_discharges, doc='Max discharge in 24 hour period')
    model.Output = Param(model.T, initialize=dict(zip(model.T.keys(), df.output.tolist())),
                         doc='original building power or post solar')

    model.DemandRates = Param(model.DemandCategories, initialize=demand_rates_dic)
    model.Vder = Param(model.T, initialize=dict(zip(list(model.T), df.vder.tolist())), doc='VDER RATES')
    model.Erate = Param(model.T, initialize=dict(zip(list(model.T), df.energy_rate.tolist())), doc='ENERGY RATES')

    # variables: Charge, discharge, binary variable, state of charge, Pmax
    model.Ein = Var(model.T, domain=NonNegativeReals, initialize=0.0)
    model.Eout = Var(model.T, domain=NonNegativeReals, initialize=0.0)
    model.X = Var(model.T, domain=NonPositiveReals, initialize=0.0)
    model.S = Var(model.T, bounds=(0, model.Smax), initialize=model.Smax)
    model.Pmax = Var(model.DemandCategories, domain=NonNegativeReals, initialize=600)

    # set itc constraint
    if itc:
        assert 'solar_output_kw' in df.columns, "can't add ITC without solar"
        model.SolarOut = Param(model.T, initialize=dict(zip(model.T.keys(), df.solar_output_kw.tolist())), doc='solar')

        def itc_constraint(model, t):
            return model.Ein[t] <= model.SolarOut[t] / .8
        model.itc_constraint = Constraint(model.T, rule=itc_constraint)

    # Pmax Constraint
    def power_constraint(model, t):
        rate_cat = df.demand_rate_category.iloc[t - 1]
        return (model.Output[t] + model.Ein[t] - model.Eout[t]) <= model.Pmax[rate_cat]

    model.power_constraint = Constraint(model.T, rule=power_constraint)

    # Storage State
    def storage_state(model, t):
        'Storage changes with flows in/out and efficiency losses'
        # Set first hour state of charge to half of max
        if t == model.T.first():
            return model.S[t] == model.Smax  # / 2 (remove the hash if you want to start with half power instead)
        else:
            return (model.S[t] == (model.S[t - 1]
                                   + dih * (model.Ein[t - 1] * np.sqrt(eff))
                                   - dih * (model.Eout[t - 1] / np.sqrt(eff))))

    model.charge_state = Constraint(model.T, rule=storage_state)

    def x_constraint(model, t):
        "Excess power that will go to the grid"
        return model.X[t] <= (model.Eout[t] - model.Output[t] - model.Ein[t])

    model.x_constraint = Constraint(model.T, rule=x_constraint)

    def discharge_constraint(model, t):
        """Maximum discharge within a single hour"""
        return model.Eout[t] <= model.Rmax

    model.discharge = Constraint(model.T, rule=discharge_constraint)

    def charge_constraint(model, t):
        """Maximum charge within a single hour"""
        return model.Ein[t] <= model.Rmax

    model.charge = Constraint(model.T, rule=charge_constraint)

    # Without a constraint the model would discharge in the final hour
    # even when SOC was 0.
    def positive_charge(model, t):
        'Limit discharge to the amount of charge in battery, including losses'
        return model.Eout[t] <= model.S[t]*np.sqrt(eff) / dih
    model.positive_charge = Constraint(model.T, rule=positive_charge)

    def daily_discharge_limit(model, t):
        """Limit on discharge within a 24 hour period"""
        max_t = model.T.last()
        num_intervals = int(24/dih)
        # Check all t until the last 24 hours
        # No need to check with < 24 hours remaining because the constraint is
        # already in place for a larger number of hours
        if t < max_t - num_intervals:
            return dih * sum(model.Eout[i] for i in range(t, t + num_intervals)) <= model.Dmax
        else:
            return Constraint.Skip
    # model.limit_out = Constraint(model.T, rule=daily_discharge_limit)

    # Define the battery income, expenses, and profit
    # Income:
    income = sum(model.Vder[t] * (model.Eout[t] - model.Ein[t] - model.Output[t] + model.X[t]) for t in model.T)
    # Expenses: Demand
    demand_expenses = sum(model.Pmax[key] * model.DemandRates[key] for key in model.Pmax.keys())

    cost = -income + demand_expenses
    model.P = cost
    model.objective = Objective(expr=cost, sense=minimize)

    # Solve the model
    solver = SolverFactory('gurobi')
    solver.solve(model)

    return vps_model_to_df(model), model


def vps_model_to_df(model):
    """
    Create a dataframe with hourly charge, discharge, state of charge, and
    price columns from a pyomo model. Only uses data from between the first
    (inclusive) and last (exclusive) hours.

    Parameters
    ----------
    model : pyomo model
        Model that has been solved

    Returns
    -------
    dataframe

    """
    # Need to increase the first & last data point by 1 because of pyomo indexing
    # and add 1 to the value of last model hour because of the range
    intervals = range(model.T.first(), model.T.last() + 1)
    Ein = [model.Ein[i].value for i in intervals]
    Eout = [model.Eout[i].value for i in intervals]
    charge_state = [model.S[i].value for i in intervals]
    X = [model.X[i].value for i in intervals]
    output = [model.Output[i] for i in intervals]
    vder = [model.Vder[t] for t in intervals]
    df_dict = dict(
        intervals=intervals,
        Ein=Ein,
        Eout=Eout,
        charge_state=charge_state,
        X=X,
        vder = vder,
        output=output,
    )

    df = pd.DataFrame(df_dict)

    return df


def vder_peak_shave_binary(df, power, capacity, max_discharges, eff=.8, project_type='storage only', itc=False):
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
    # df = df.loc[first_model_hour:last_model_hour, :]
    df = filter_data(df, project_type)

    # dih: data interval hours
    dih = (df.date.iloc[1] - df.date.iloc[0])
    dih = dih.total_seconds() / 3600
    dih = abs(dih)

    # TODO check back this function
    demand_categories, demand_rates_dic = filter_demand_rates(df)

    # Define model sets and parameters
    model = ConcreteModel()
    model.T = Set(doc='index', initialize=(df.index + 1).tolist(), ordered=True)
    model.DemandCategories = Set(initialize=demand_categories, doc='demand rate categories', ordered=True)
    model.Rmax = Param(initialize=power, doc='Max rate of power flow (kW) in or out')
    model.Smax = Param(initialize=capacity, doc='Max storage (kWh)')
    model.Dmax = Param(initialize=capacity * max_discharges, doc='Max discharge in 24 hour period')
    model.Output = Param(model.T, initialize=dict(zip(model.T.keys(), df.output.tolist())),
                         doc='original building power or post solar')

    model.DemandRates = Param(model.DemandCategories, initialize=demand_rates_dic)
    model.Vder = Param(model.T, initialize=dict(zip(list(model.T), df.vder.tolist())), doc='VDER RATES')
    model.Erate = Param(model.T, initialize=dict(zip(list(model.T), df.energy_rate.tolist())), doc='ENERGY RATES')

    # variables: Charge, discharge, binary variable, state of charge, Pmax
    model.Ein = Var(model.T, domain=NonNegativeReals, initialize=0.0)
    model.Eout = Var(model.T, domain=NonNegativeReals, initialize=0.0)
    model.Ev_out = Var(model.T, domain=NonNegativeReals, initialize=0.0)
    model.X = Var(model.T, domain=Binary, initialize=0.0)
    model.S = Var(model.T, bounds=(0, model.Smax), initialize=model.Smax)
    model.Pmax = Var(model.DemandCategories, domain=NonNegativeReals, initialize=600)

    # set itc constraint
    if itc:
        assert 'solar_output_kw' in df.columns, "can't add ITC without solar"
        model.SolarOut = Param(model.T, initialize=dict(zip(model.T.keys(), df.solar_output_kw.tolist())), doc='solar')

        def itc_constraint(model, t):
            return model.Ein[t] <= model.SolarOut[t] / .8
        model.itc_constraint = Constraint(model.T, rule=itc_constraint)

    # Pmax Constraint
    def power_constraint(model, t):
        rate_cat = df.demand_rate_category.iloc[t - 1]
        return (model.Output[t] + model.Ein[t] - model.Eout[t]) <= model.Pmax[rate_cat]

    model.power_constraint = Constraint(model.T, rule=power_constraint)

    def vder_limit(model, t):
        if model.Rmax <= model.Output[t]:
            return model.Ev_out[t] == 0
        else:
            return model.Ev_out[t] <= model.Rmax - model.Output[t]
    model.vder_limit = Constraint(model.T, rule=vder_limit)

    def binary(model, t):
        return model.X[t] >= model.Ev_out[t]/model.Rmax

    model.binary = Constraint(model.T, rule=binary)

    def ein(model, t):
        return model.Ein[t] <= model.Rmax * (1-model.X[t])

    model.ein = Constraint(model.T, rule=ein)

    def eout(model,t):
        return model.Eout[t] <= model.Rmax

    model.eout = Constraint(model.T, rule=eout)

    # Storage State
    def storage_state(model, t):
        'Storage changes with flows in/out and efficiency losses'
        # Set first hour state of charge to half of max
        if t == model.T.first():
            return model.S[t] == model.Smax  # / 2 (remove the hash if you want to start with half power instead)
        else:
            return (model.S[t] == (model.S[t - 1]
                                   + dih * (model.Ein[t - 1] * np.sqrt(eff))
                                   - dih * (model.Eout[t - 1] / np.sqrt(eff))
                                   - dih * model.Ev_out[t-1] / np.sqrt(eff)
                                   - dih * model.Output[t-1]*model.X[t] / np.sqrt(eff)))

    model.charge_state = Constraint(model.T, rule=storage_state)

    # Without a constraint the model would discharge in the final hour
    # even when SOC was 0.
    def positive_charge(model, t):
        'Limit discharge to the amount of charge in battery, including losses'
        return model.Eout[t] <= model.S[t]*np.sqrt(eff) / dih
    model.positive_charge = Constraint(model.T, rule=positive_charge)

    def daily_discharge_limit(model, t):
        """Limit on discharge within a 24 hour period"""
        max_t = model.T.last()
        num_intervals = int(24/dih)
        # Check all t until the last 24 hours
        # No need to check with < 24 hours remaining because the constraint is
        # already in place for a larger number of hours
        if t < max_t - num_intervals:
            return dih * sum(model.Eout[i] for i in range(t, t + num_intervals)) <= model.Dmax
        else:
            return Constraint.Skip
    # model.limit_out = Constraint(model.T, rule=daily_discharge_limit)

    # Define the battery income, expenses, and profit
    # Income:
    income = sum(model.Vder[t] * model.Ev_out[t] for t in model.T)
    # Expenses: Demand
    demand_expenses = sum(model.Pmax[key] * model.DemandRates[key] for key in model.Pmax.keys())

    cost = income - demand_expenses
    model.P = cost
    model.objective = Objective(expr=cost, sense=maximize)

    # Solve the model
    solver = SolverFactory('gurobi')
    solver.solve(model)

    return vpsb_model_to_df(model), model


def vpsb_model_to_df(model):
    """
    Create a dataframe with hourly charge, discharge, state of charge, and
    price columns from a pyomo model. Only uses data from between the first
    (inclusive) and last (exclusive) hours.

    Parameters
    ----------
    model : pyomo model
        Model that has been solved

    Returns
    -------
    dataframe

    """
    # Need to increase the first & last data point by 1 because of pyomo indexing
    # and add 1 to the value of last model hour because of the range
    intervals = range(model.T.first(), model.T.last() + 1)
    Ein = [model.Ein[i].value for i in intervals]
    Eout = [model.Eout[i].value for i in intervals]
    Ev_out = [model.Ev_out[t].value for t in intervals]
    charge_state = [model.S[i].value for i in intervals]
    X = [model.X[i].value for i in intervals]
    output = [model.Output[i] for i in intervals]
    vder = [model.Vder[t] for t in intervals]
    df_dict = dict(
        intervals=intervals,
        Ein=Ein,
        Eout=Eout,
        Ev_out=Ev_out,
        charge_state=charge_state,
        X=X,
        vder=vder,
        output=output,
    )

    df = pd.DataFrame(df_dict)

    return df



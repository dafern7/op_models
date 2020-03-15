import numpy as np
from pyomo.environ import *
import pandas as pd
import helper
import os
from op_helpers import filter_data, filter_demand_rates

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
    model.Qplus = Var(model.T, domain=NonNegativeReals, initialize=0)
    model.Qnet = Exception(expr=model.Output + model.Ein - model.Eout)

    # set itc constraint
    if itc:
        assert 'solar_output_kw' in df.columns, "can't add ITC without solar"
        model.SolarOut = Param(model.T, initialize=dict(zip(model.T.keys(), df.solar_output_kw.tolist())), doc='solar')

        def itc_constraint(model, t):
            return model.Ein[t] <= model.SolarOut[t] / .8
        model.itc_constraint = Constraint(model.T, rule=itc_constraint)

    #QNet:
    def q_constraint(model, t):
        return model.Qnet[t] <= model.Qplus[t]

    model.q_constraint = Constraint(model.T, rule=q_constraint)

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
            return model.S[t] == model.Smax  # / 2 (remove the hash if you want to start with half power instead)
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
    income = sum(model.Vder[t] * (model.Qnet[t] - model.Qplus[t]) for t in model.T)
    # Expenses: Energy:
    energy_expenses = dih * sum(model.Erate[t] * model.Qplus[t] for t in model.T)
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

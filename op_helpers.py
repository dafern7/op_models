import pandas as pd
from functools import reduce
import helper
import os

def model_to_df(model, project_type):
    """
    Create a dataframe with hourly charge, discharge, state of charge, and
    output columns from a pyomo model.

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
    date = [model.date[i] for i in intervals]
    Ein = [model.Ein[i].value for i in intervals]
    date = [model.date[i] for i in intervals]
    Eout = [model.Eout[i].value for i in intervals]
    charge_state = [model.S[i].value for i in intervals]
    #output = [model.D[i] for i in intervals]
    if project_type == 'solar+storage':
        solar = [model.Solar[i] for i in intervals]
    else:
        solar = [0 for i in intervals]
    system_out = [(model.Eout[i].value + model.Solar[i] - model.Ein[i].value) for i in intervals]
    df_dict = dict(
        intervals=intervals,
        solar=solar,
        #output=output,
        date = date,
        Ein=Ein,
        Eout=Eout,
        charge_state=charge_state,
        system_out=system_out,
    )

    df = pd.DataFrame(df_dict)

    return df


# Demand Rate finder
def filter_demand_rates(df, num_columns=1):
    if num_columns >1:
        itr1 = ['demand_rate_category' + str(i) for i in range(1, num_columns+1)]
        itr2 = ['demand_rate' + str(i) for i in range(1, num_columns+1)]
        dics = {}
        cats = []
        for i1, i2 in zip(itr1, itr2):
            rate_categories = []
            dic = {}
            for rate_cat in set(df[i1]):
                rate_categories.append(rate_cat)
                dic[rate_cat] = df.loc[df[i1] == rate_cat, i2].values[0]
            dics.update(dic)
            cats.append(rate_categories)
        cats = reduce(lambda x,y: x+y,cats)        
        return cats, dics
    else:
        rate_categories = []
        dic = {}
        for rate_cat in set(df.demand_rate_category):
            rate_categories.append(rate_cat)
            dic[rate_cat] = df.loc[df.demand_rate_category == rate_cat, 'demand_rate'].values[0]
        return rate_categories, dic


# lsrv filter
def filter_lsrv_df(df):
    assert 'lsrv_rate' in df.columns, 'lsrv_rate is not found in columns'
    assert 'lsrv_event' in df.columns, 'lsrv_event is not found in columns'
    lsrv_events = []
    dic = {}
    for event in set(df.lsrv_event):
        lsrv_events.append(event)
        dic[event] = df.loc[df.lsrv_event == event, 'lsrv_rate'].values[0]
    return lsrv_events, dic


# Filter data depending on project type and add vder values based on derv,icap3, and lbmpm
def filter_data(df, project_type):
    if 'original_building_power_kw' not in df.columns:
        df['original_building_power_kw'] = 0

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

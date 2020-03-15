from sklearn.utils import Bunch
import pickle
import os
import warnings
from os.path import join
import pandas as pd
import requests
import numpy as np
import copy
import matplotlib.pyplot as plt
import calendar
import datetime
import matplotlib
from calculator import Calculator
from contextlib import contextmanager
import sys, os
from calendar import monthrange
import op_models


def save_data(data, file_name=None, overwrite=False, description=None):
    """

    :param data: data that you want to save
    :param file_name: str, name of the file you want to save
    :param overwrite: bool, whether you want to overwrite an existing file saved previously
    :param description: str, description of the data you want to save
    :return: None
    """
    assert file_name, 'file_name is not provided'
    # assert os.path.isdir("./datasets"), "wrong working directory, datasets folder doesn't exist"
    if (not overwrite) & os.path.exists(join("./datasets", file_name + '.p')):
        raise AssertionError('you have the same file name, set overwrite to True or rename your file')
    if not description:
        warnings.warn('No description is provided')

    save = Bunch(data=data, description=description)
    directory = os.path.dirname(os.path.abspath(__file__))
    directory = join(directory, "datasets", file_name + '.p')
    pickle.dump(save, open(directory, 'wb'))


def load_data(file_name, verbose=False):
    """

    :param file_name: str, file name you want to retrieve
    :param verbose: if True prints description of the data retrieved
    :return: the saved data

    """
    assert isinstance(file_name, str), 'file_name is not a string'
    if file_name[-2:] != '.p':
        file_name += '.p'
    directory = os.path.dirname(os.path.abspath(__file__))
    directory = join(directory, 'datasets', file_name)
    if verbose:
        print(pickle.load(open(directory, mode='rb')).description)
    return pickle.load(open(directory, mode='rb')).data


def pv_data(coordinates=None, address=None, tilt=30,
            solar_size_kw_dc=4500, inverter_size_kw=4500, inverter_efficiency=96,
            system_losses_perc_of_dc_energy=14,
            mount="Fixed (open rack)", module_type="Default [Standard]"):
    """
    :param coordinates: optional, set the coordinates of the location as (latitude, longitude)
    :param address: optional, set the address as a string
    :param tilt: integer, tilt angle default = 40
    :param solar_size_kw_dc: solar size in KW
    :param inverter_size_kw: inverter size in KW
    :param inverter_efficiency: inverter efficiency, default = 96 (96%)
    :param system_losses_perc_of_dc_energy: system losses, default = 14 (14%)
    :param mount: mount type, default = 'Fixed (open rack)'
    :param module_type: module type, default= "Default [Standard]"
    :return: hourly data of expected solar energy
    """
    # TODO:  Figure why inverter_size_kw, export_limit_ac_k_w, export_limit_ac_k_w, dataset,
    #  and annual_solar_degradation_perc_per_year are not implemented
    if coordinates:
        assert isinstance(coordinates, tuple), 'coordinates should be in tuple format'
        latitude = coordinates[0]
        longitude = coordinates[1]
        location = "&" + "lat=" + str(latitude) + "&" + "lon=" + str(longitude)
        tilt = latitude
        verbose_dic = {'Coordinates': coordinates, 'Tilt Angle': int(tilt)}
    elif address:
        location = "&" + "address=" + address
        verbose_dic = {'Address': address, 'Tilt Angle': int(tilt)}
    else:
        raise Exception('either coordinates or address must be input')
    verbose_dic.update({'Solar System Size (kW DC)': solar_size_kw_dc, 'Inverter Size (kW)': inverter_size_kw,
                        'Inverter Efficiency': inverter_efficiency,
                        'System Losses Percentage of DC Energy': system_losses_perc_of_dc_energy})
    save_data(verbose_dic, 'solar_inputs', overwrite=True, description='Solar Inputs to View on Excel')
    # Solar Input Selection Setup and API_Key Setup
    pv_watts_api_key = "VTF48OxZfq7tlP4oriEDVK1qAnpOCPdzl0XGT2c0"
    mount_dict = {"Fixed (open rack)": 0, "Fixed (roof mount)": 1, "1-Axis Tracking": 2,
                  "1-Axis Backtracking": 3, "2-Axis": 4, "Default [Fixed (open rack)]": 0}
    module_type_dict = {"Standard": 0, "Premium": 1, "Thin film": 2, "Default [Standard]": 0}
    array_azimuth_dict = {"135 (SE)": 135, "180 (S)": 180, "225 (SW)": 225}
    dataset = "nsrdb"

    array_azimuth = array_azimuth_dict["180 (S)"]
    mount = mount_dict[mount]
    module_type = module_type_dict[module_type]

    # Over a 25 year life
    annual_solar_degradation_perc_per_year = 0.5 * .01
    # If no export limit then state "No Limit"
    export_limit_ac_k_w = 3000
    variable = 1

    get_link = ''.join(["https://developer.nrel.gov/api/pvwatts/v6.json?" + "api_key=" + pv_watts_api_key,
                        "&" + "system_capacity=" + str(solar_size_kw_dc),
                        "&" + "module_type=" + str(module_type),
                        "&" + "losses=" + str(system_losses_perc_of_dc_energy),
                        "&" + "array_type=" + str(mount),
                        "&" + "tilt=" + str(tilt),
                        "&" + "azimuth=" + str(array_azimuth),
                        location,
                        "&" + "dataset=nsrdb",
                        "&" + "timeframe=hourly",
                        "&" + "inv_eff=" + str(inverter_efficiency), ]
                       )
    result = requests.get(get_link)
    data = result.json()
    outs_w = data["outputs"]["ac"]
    outs_kw = [out / 1000 for out in outs_w]
    outs_kw = pd.Series(outs_kw)
    df = load_data('hours_in_year')
    df['solar_output_kw'] = outs_kw.values

    return df


def add_pv_to_df(data, solar, net_metering=True,
                 power_data='original_building_power_kw', energy_data='original_building_energy_kwh'):
    """

    :param net_metering: set True if net metering is available
    :param data: Data frame, must contain the columns ['date, hour']
    :param solar: Data frame, must contain the following columns:
    - solar_output_kw : AC output power from the solar system in kw
    - hour : hours
    - day : days
    - month : months
    :param power_data: original power data in kw
    :param energy_data: original energy data in kwh
    :return: data frame with solar_output_kw column
    """
    # assert 'hour' in data.columns, 'data frame has no hour column'
    # assert isinstance(data.date, pd.datetime), 'date column is not a datetime data type'
    assert 'date' in data.columns, 'data frame has no date column'
    assert 'solar_output_kw' in solar.columns, 'solar data has no solar output named "solar_output_kw" please add or' \
                                               ' rename'
    assert 'hour' in solar.columns, 'solar data has no hour column'
    assert 'day' in solar.columns, 'solar data has no date column'
    assert 'month' in solar.columns, 'solar data has no month column'

    data = copy.deepcopy(data)
    data['hour'] = data.date.dt.hour
    data['solar_output_kw'] = 0
    pd.options.mode.chained_assignment = None
    frames = []
    for month in range(1, 13):
        solar_month = solar.loc[solar.month == month]
        data_month = data.loc[data.date.dt.month == month]
        for hour in range(0, 24):
            solar_month_hour = solar_month.loc[solar_month.hour == hour]
            data_month_hour = data_month.loc[data_month.hour == hour]
            for day in solar_month_hour.day:
                data_mhd = data_month_hour.loc[data_month_hour.date.dt.day == day]
                data_mhd.solar_output_kw = solar_month_hour.solar_output_kw.loc[solar_month_hour.day == day].values[0]
                frames.append(data_mhd)
    df = pd.concat(frames, sort=True).sort_index()
    df.drop('hour', axis=1, inplace=True)
    data_interval_hrs = (df.date.iloc[1] - df.date.iloc[0])
    data_interval_hrs = data_interval_hrs.total_seconds() / 3600
    dih = abs(data_interval_hrs)

    x = df[power_data] - df['solar_output_kw']
    if net_metering:
        df['power_post_solar_kw'] = x * (x >= 0)
        for idx, val in enumerate(x):
            if val < 0:
                df[power_data].iloc[idx] = df['solar_output_kw'].iloc[idx]
                df['power_post_solar_kw'].iloc[idx] = 0
                df[energy_data].iloc[idx] = df[power_data].iloc[idx] * dih
    else:
        df['power_post_solar_kw'] = x
    df['energy_post_solar_kwh'] = dih * df['power_post_solar_kw']
    return df


def fast_add_pv_to_df(df, solar):
    frames = []
    for year in set(df.date.dt.year):
        df_year = df.loc[df.date.dt.year == year]
        df_year.drop_duplicates(subset=['date'], inplace=True)
        df_year.sort_values('date', inplace=True)
        start = df_year.iloc[0]
        end = df_year.iloc[-1]
        cond_s = (solar.month == start.date.month) & (solar.hour == start.date.hour) & (solar.day == start.date.day)
        cond_e = (solar.month == end.date.month) & (solar.hour == end.date.hour) & (solar.day == end.date.day)
        start_idx = solar[cond_s].index[0]
        end_idx = solar[cond_e].index[0]
        solar_to = solar.iloc[start_idx:end_idx + 1]
        solar_to = solar_to.iloc[np.repeat(np.arange(len(solar_to)), 4)]
        df_year['solar_output_kw'] = solar_to['solar_output_kw'].values
        frames.append(df_year)
    df = pd.concat(frames)

    x = df['original_building_power_kw'] - df['solar_output_kw']

    df['power_post_solar_kw'] = x * (x >= 0)

    for idx, val in enumerate(x):
        if val <= 0:
            df['original_building_power_kw'].iloc[idx] = df['solar_output_kw'].iloc[idx]
            df['original_building_energy_kwh'].iloc[idx] = df['original_building_power_kw'].iloc[idx] / 4
            df['power_post_solar_kw'].iloc[idx] = 0

    data_interval_hrs = (df.date.iloc[1] - df.date.iloc[0])
    data_interval_hrs = data_interval_hrs.total_seconds() / 3600
    df['energy_post_solar_kwh'] = data_interval_hrs * df['power_post_solar_kw']
    return df


def add_vder_to_df(data, vder):
    """

    """
    # assert 'hour' in data.columns, 'data frame has no hour column'
    # assert isinstance(data.date, pd.datetime), 'date column is not a datetime data type'
    assert 'date' in data.columns, 'data frame has no date column'

    assert 'hour' in vder.columns, 'solar data has no hour column'
    assert 'day_of_year' in vder.columns, 'solar data has no day_of_year column'
    assert 'month' in vder.columns, 'solar data has no month column'

    data = copy.deepcopy(data)
    data['hour'] = data.date.dt.hour
    data['vder'] = 0
    # data interval hours
    dih = (data.date.iloc[1] - data.date.iloc[0])
    dih = dih.total_seconds() / 3600
    dih = abs(dih)
    # set assignment warning off
    pd.options.mode.chained_assignment = None
    frames = []
    for month in range(1, 13):
        vder_month = vder.loc[vder.month == month]
        data_month = data.loc[data.date.dt.month == month]
        for hour in range(0, 24):
            vder_month_hour = vder_month.loc[vder_month.hour == hour]
            data_month_hour = data_month.loc[data_month.hour == hour]
            for day in vder_month_hour.day:
                data_mhd = data_month_hour.loc[data_month_hour.date.dt.day == day]
                data_mhd.vder = vder_month_hour.total.loc[vder_month_hour.day == day].values[0] * dih
                frames.append(data_mhd)
    df = pd.concat(frames, sort=True).sort_index()
    df.drop('hour', axis=1, inplace=True)
    return df


def predict_price(energy, power):
    """
    :param energy: capacity of the battery in kwh
    :param power: power of the battery in kw
    :return: predicted price of the whole system
    """
    from scipy import interpolate
    df = load_data('battery_price_list')
    assert power in set(df.power_kw), 'available power values are {}'.format(sorted(set(df.power_kw)))
    dfp = df[df.power_kw == power]
    x = dfp.capacity_kwh
    b = interpolate.interp1d(x, dfp.battery_price, fill_value="extrapolate")
    c = interpolate.interp1d(x, dfp.commissioning, fill_value="extrapolate")
    s = interpolate.interp1d(x, dfp.shipping, fill_value="extrapolate")
    battery = b(energy)
    commission = c(energy)
    shipping = s(energy)
    total = 1.2 * (battery * 1.0925 + commission + shipping + 10000)
    return total


def df_interval_hours(df):
    dih = (df.date.iloc[1] - df.date.iloc[0])
    dih = dih.total_seconds() / 3600
    return abs(dih)


def find_first_error(dataframe, replsolar):
    """
    :param dataframe: input dataframe
    :param replsolar: dataframe with a 'solar' column to be appended to the input dataframe
    :return newt: the input dataframe up to the first error/inconsistency
    :return FirstFalse: the index of the first error/inconsistency (mismatch with solar data)
    """

    dataframe = copy.deepcopy(dataframe)
    months = list(dataframe.date.dt.month.values)
    replsolar_months = list(replsolar.month.values)
    days = list(dataframe.date.dt.day.values)
    replsolar_days = list(replsolar.day.values)
    hours = list(dataframe.date.dt.hour.values)
    replsolar_hours = list(replsolar.hour.values)
    start = 0
    n = len(dataframe)
    bools = np.equal(np.equal(months[start:n], replsolar_months[start:n]),
                     np.equal(hours[start:n], replsolar_hours[start:n]),
                     np.equal(days[start:n], replsolar_days[start:n]))
    boolslist = list(bools)
    if False in boolslist:
        FirstFalse = boolslist.index(False)
    else:
        FirstFalse = len(dataframe)

    newt = dataframe.drop(dataframe.index[FirstFalse:])
    newt['solar_output_kw'] = replsolar.solar_output_kw[start:FirstFalse].values
    return newt, FirstFalse


def remove_solar_values(dataframe, FirstFalse, replicated_solar, start):
    """
    :param dataframe: input dataframe should be the same as the find first error one
    :param FirstFalse: from find_first_error
    :param replicated_solar: the dataframe usually has four data points per hour, solar only has one, need to replicate solar to match
    :param start: counter for the index of the first False to appear in the dataset. Should end up at the last index if there are no inconsistencies
    :return new_replsolar: replicated solar dataframe with removed values
    :return summedFalse: counter for the first False
    :return remaining_t: the rest of the input dataframe after the first inconsistency
    """
    dataframe = copy.deepcopy(dataframe)
    summedFalse = start + FirstFalse
    remaining_t = dataframe.loc[dataframe.index >= summedFalse]  # [summedFalse::]
    startmo = remaining_t.date.dt.month[summedFalse]
    startday = remaining_t.date.dt.day[summedFalse]
    starthr = remaining_t.date.dt.hour[summedFalse]
    boolsindex = list(
        (replicated_solar.month == startmo) & (replicated_solar.day == startday) & (replicated_solar.hour == starthr))
    startindex = boolsindex.index(True)
    new_replsolar = replicated_solar.drop(replicated_solar.index[0:startindex])
    return new_replsolar, summedFalse, remaining_t


# takes a dataframe and reformats it so January is first listing
def reformat(dataframe):
    dataframe = copy.deepcopy(dataframe)
    dataframe['month'] = dataframe.date.dt.month
    dataframe['day'] = dataframe.date.dt.day
    dataframe['hour'] = dataframe.date.dt.hour
    dataframe.sort_values(by=['month', 'day', 'hour'], inplace=True)
    dataframe = dataframe.reset_index(drop=True)
    dataframe.drop(columns=['month', 'day', 'hour'], inplace=True)
    return dataframe


def add_solar_to_df(dataframe, solar, hourly_intervals):
    """
    :param dataframe: input dataframe
    :param solar input dataframe with solar column to be appended to input dataframe
    :param hourly_intervals
    :return df: dataframe with appended solar column
    """
    assert 'date' in dataframe.columns, 'dataframe has no date column'
    start_index = dataframe.loc[dataframe.date.dt.hour == 0].index[0]
    cut_dataframe = dataframe.drop(dataframe.index[0:start_index])
    new = reformat(cut_dataframe)
    iterated_var = copy.deepcopy(new)
    repl_solar = pd.DataFrame(pd.np.repeat(solar.values, hourly_intervals, axis=0), columns=solar.columns)
    false_count = 0
    df = pd.DataFrame(columns=dataframe.columns.tolist() + ['solar_output_kw'])
    while True:
        new_var, iter_num_false = find_first_error(iterated_var, repl_solar)
        try:
            new_replsolar, second_num_false, remaining_iter_var = remove_solar_values(iterated_var, iter_num_false,
                                                                                      repl_solar, false_count)
        except IndexError:
            df = pd.concat([df, new_var], sort=False)  # may be some issues when update occurs
            break
        false_count = second_num_false
        repl_solar = new_replsolar
        iterated_var = remaining_iter_var
        df = pd.concat([df, new_var], sort=False)
    df['power_post_solar_kw'] = df['original_building_power_kw'] - df['solar_output_kw']
    df['energy_post_solar_kwh'] = df['power_post_solar_kw'] / hourly_intervals
    return df


# plot daily peak load per month:
class PlotLoad:
    def __init__(self, dataframe, power_data='original_building_power_kw'):
        self.df = copy.deepcopy(dataframe)
        self.df.reset_index(drop=True, inplace=True)
        self.pd = power_data

    def by_month(self, month, year):
        dfy = copy.deepcopy(self.df[self.df.date.dt.year == year])
        dic_max = {}
        dic_mean = {}
        for each_month in set(dfy.date.dt.month):
            df_m = dfy[dfy.date.dt.month == each_month]
            for day in set(df_m.date.dt.day):
                dic_max[(each_month, day)] = df_m[self.pd][df_m.date.dt.day == day].max()
                dic_mean[(each_month, day)] = df_m[self.pd][df_m.date.dt.day == day].mean()
        d = [val for val, key in zip(dic_max.values(), dic_max.keys()) if key[0] == month]
        m = [val for val, key in zip(dic_mean.values(), dic_mean.keys()) if key[0] == month]
        plt.grid()
        plt.step(list(range(1, len(d) + 1)), d, label='daily_peak')
        plt.step(list(range(1, len(m) + 1)), m, label='daily_avg')
        plt.xlabel('day of month')
        plt.ylabel('power (kW)')
        plt.title('daily peaks & daily averages for {} {}'.format(calendar.month_name[month], year))
        plt.legend()
        plt.show()

    def by_year(self, year):
        dfy = copy.deepcopy(self.df[self.df.date.dt.year == year])
        dic_mean = {}
        for each_day in set(dfy.date.dt.dayofyear):
            df_d = dfy[dfy.date.dt.dayofyear == each_day]
            dic_mean[each_day] = df_d[self.pd].mean()
        days = list(dic_mean.keys())
        powers = list(dic_mean.values())
        plt.step(days, powers, label='daily_load')
        plt.grid()
        plt.xlabel('Month')
        plt.xticks([1] + [30 * i for i in range(1, 12)], calendar.month_abbr[1:])
        plt.ylabel('power (kW)')
        plt.title('daily load profile for {}'.format(year))
        plt.show()

    def all(self):
        dic_mean = {}
        for each_year in set(self.df.date.dt.year):
            dfy = self.df[self.df.date.dt.year == each_year]
            for each_day in set(dfy.date.dt.dayofyear):
                df_d = dfy[dfy.date.dt.dayofyear == each_day]
                dic_mean[(each_year, each_day)] = df_d[self.pd].mean()
        days = [str(day) + '-' + str(year) for year, day in dic_mean.keys()]
        days = [datetime.datetime.strptime(day, '%j-%Y') for day in days]
        powers = list(dic_mean.values())
        plt.plot(days, powers, label='daily_load')
        plt.grid()
        plt.xlabel('month')
        a = [day for day in days if day.day == 1]
        b = [day.month for day in a]
        plt.xticks(a, b)
        plt.ylabel('power (kW)')
        years = list(set(self.df.date.dt.year))
        plt.title('daily load profile from {} to {}'.format(years[0], years[-1]))
        plt.show()

    def savings_per_month(self, min_tick, max_tick, tick_spacing):
        dic_original = {}
        for month in set(self.df.date.dt.month):
            dic_original[month] = 0
            sfm = self.df[self.df.date.dt.month == month]
            dic_original[month] += (sfm.original_building_energy_kwh * sfm.energy_rate).sum()
            for rate_category in set(sfm.demand_rate_category):
                sfmr = sfm[sfm.demand_rate_category == rate_category]
                dic_original[month] += sfmr.original_building_power_kw.max() * sfmr.iloc[0].demand_rate

        dic_post = {}
        for month in set(self.df.date.dt.month):
            dic_post[month] = 0
            sfm = self.df[self.df.date.dt.month == month]
            dic_post[month] += (sfm.energy_post_storage_kwh * sfm.energy_rate).sum()
            for rate_category in set(sfm.demand_rate_category):
                sfmr = sfm[sfm.demand_rate_category == rate_category]
                dic_post[month] += sfmr.power_post_storage_kw.max() * sfmr.iloc[0].demand_rate
            if 'vder' in self.df.columns:
                dic_post[month] += (sfmr.vder * sfmr.loc[sfmr.power_post_storage_kw < 0, 'power_post_storage_kw']).sum()
        dic_net = {}
        for key in dic_original.keys():
            dic_net[key] = dic_original[key] - dic_post[key]

        y = [i for i in range(min_tick, max_tick, tick_spacing)]
        y_ticks = [str(int(tick / 1000)) + ',000' for tick in y]
        plt.bar(list(dic_original.keys()), list(dic_original.values()), color='c', label='Originial Cost')
        plt.bar(list(dic_post.keys()), list(dic_post.values()), color='lightblue', label='After Solar+Storage')
        plt.legend()
        plt.xlabel('Month')
        plt.ylabel('Cost $')
        plt.title('Total Savings Per Month')
        plt.yticks(y, y_ticks)
        plt.xticks(list(dic_original.keys()))
        plt.grid(alpha=.5, axis='y')
        # manager = plt.get_current_fig_manager()
        # manager.frame.Maximize(True)
        # plt.savefig('Total Savings Per Month.png')
        plt.savefig('Total Savings Per Month.pdf', bbox_inches='tight', pad_inches=.4)
        plt.show()


# Separated from PlotLoad
def generate_histogram(dict1, dict2):
    """
    :param dict1: initial dictionary, stacked on the bottom
    :param dict2: stacked on top
    :return: stacked histogram
    dict1 is the monthly estimated cost & month2 is the estimated cost after project is installed.
    """

    original_keys = list(dict1.keys())
    original_val = list(dict1.values())

    p_original = plt.bar(original_keys, original_val, 0.6, color='lightblue', )
    p_post = plt.bar(list(dict2.keys()), list(dict2.values()), 0.6, bottom=original_val, color='blue')

    ax = plt.axes()
    ax.get_yaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda y, p: format(int(y), ',')))
    ax.get_yaxis().set_major_formatter(
        matplotlib.ticker.FormatStrFormatter('$%d'))
    plt.xticks(list(dict1.keys()))
    ax.grid(axis='y', alpha=.3)

    plt.ylabel('Cost')
    plt.xlabel('Months')
    plt.legend((p_original[0], p_post[0]), ('Original Cost', 'Post Solar Cost'))

    plt.show()

    # need to preprocess data before using functions - data must have only 'date' and 'original_building_power_kw'


def use_peak_shave_model(df, solar, hourly_intervals, tariff_id, power, capacity, eff=0.9, itc=True,
                         project_type='solar+storage', export=False):
    """
    :param df: input dataframe, make sure it has 'date' and 'original_building_power_kw columns
    :param solar: solar dataframe
    :param hourly_intervals: how many data points in an hour
    :param tariff_id: demand/energy tariff id
    :param power: power of system
    :param capacity: capacity of system
    :param eff: efficiency of system
    :param itc: itc incentive eligibility
    :param project_type: solar+storage or storage only
    :param export: able to export power to grid
    :return relevant_data5: dataframe with power post solar and storage calculations
    """

    assert 'original_building_power_kw' in df.columns, 'original_building_power_kw must be in columns'
    assert isinstance(df.date.iloc[0], datetime.datetime), 'date is not datetime format'

    # if original_building energy kwh isn't in the df, then put it in
    df['original_building_energy_kwh'] = df.original_building_power_kw / hourly_intervals
    relevant_data4 = Calculator(df, tariff_id)
    relevant_data4.add_solar(solar, hourly_intervals, mode='fast')
    relevant_data5 = relevant_data4.solar_data_transform['dataframe']
    modelled, model = op_models.peak_shave(relevant_data5, power, capacity, eff, itc, project_type, export)
    relevant_data5['power_post_solarandstorage_kw'] = modelled.system_out
    relevant_data5['energy_post_solarandstorage_kwh'] = modelled.system_out / hourly_intervals
    return relevant_data5


def annual_savings(relevant_data5, frac_of_year=1, verbose=False, project_type='solar+storage'):
    """
    :param relevant_data5: dataframe output from use_peak_shave_model func format
    :param frac_of_year: fraction of the year available in data (3mo of data = 1/4)
    :param project_type: solar+storage or storage_only
    :return general_building_cost: sum of building costs
    :return general_cost_after_all: sum of costs after solar and storage
    :return gen_cost_after_solar: sum of costs after solar only
    """
    assert 'energy_post_solarandstorage_kwh' in relevant_data5.columns, 'you should run peak_shave_model first'

    est_build_energy_cost = sum(relevant_data5.original_building_energy_kwh * relevant_data5.energy_rate)
    eng_cost_after_solar = sum(relevant_data5.energy_post_solar_kwh * relevant_data5.energy_rate)
    # TODO: energy_post_solarandstorage_kwh (Same thing if it was storage only)
    eng_cost_after_all = sum(relevant_data5.energy_post_solarandstorage_kwh * relevant_data5.energy_rate)
    eng_cost_storage = eng_cost_after_all - eng_cost_after_solar

    energy_savings_total = est_build_energy_cost - eng_cost_after_all
    energy_savings_solar = energy_savings_total * (eng_cost_after_solar / (eng_cost_storage + eng_cost_after_solar))
    energy_savings_storage = energy_savings_total * (eng_cost_storage / (eng_cost_storage + eng_cost_after_solar))

    percent_savings = (energy_savings_total / est_build_energy_cost) * 100
    if verbose:
        if project_type == 'solar+storage':
            print('Estimated Building Energy Costs: ${:0.2f}'.format(est_build_energy_cost * (1 / frac_of_year)))
            print('Estimated Energy Cost After Solar+Storage: ${:0.2f}'.format(eng_cost_after_all * (1 / frac_of_year)))
            print('Estimated Energy Savings per year: ${:0.2f}'.format(energy_savings_total * (1 / frac_of_year)),
                  'Solar portion: ${:0.2f}'.format(energy_savings_solar * (1 / frac_of_year)),
                  'Storage portion: ${:0.2f}'.format(energy_savings_storage * (1 / frac_of_year)))
            print("Percent Energy Savings: {:0.2f}% \n".format(percent_savings))
        else:
            print('Estimated Building Energy Costs: ${:0.2f}'.format(est_build_energy_cost * (1 / frac_of_year)))
            print('Estimated Energy Cost After Storage Only: ${:0.2f}'.format(eng_cost_after_all * (1 / frac_of_year)))
            print('Estimated Energy Savings per year: ${:0.2f}'.format(energy_savings_total * (1 / frac_of_year)))
            print("Percent Energy Savings: {:0.2f}% \n".format(percent_savings))

    est_build_demand_cost, dem_cost_after_solar, dem_cost_after_all = get_demand_cost(relevant_data5)
    dem_cost_storage = dem_cost_after_all - dem_cost_after_solar

    demand_savings_total = est_build_demand_cost - dem_cost_after_all
    demand_savings_solar = demand_savings_total * (dem_cost_after_solar / (dem_cost_storage + dem_cost_after_solar))
    demand_savings_storage = demand_savings_total * (dem_cost_storage / (dem_cost_storage + dem_cost_after_solar))

    percent_dem_savings = (demand_savings_total / est_build_demand_cost) * 100

    if verbose:
        if project_type == 'solar+storage':
            print('Estimated Building Demand Costs: ${:0.2f}'.format(est_build_demand_cost * (1 / frac_of_year)))
            print('Estimated Demand Cost After Solar+Storage: ${:0.2f}'.format(dem_cost_after_all * (1 / frac_of_year)))
            print('Estimated Demand Savings per year: ${:0.2f}'.format(demand_savings_total * (1 / frac_of_year)),
                  'Solar portion: ${:0.2f}'.format(demand_savings_solar * (1 / frac_of_year)),
                  'Storage portion: ${:0.2f}'.format(demand_savings_storage * (1 / frac_of_year)))
            print("Percent Demand Savings: {:0.2f}% \n".format(percent_dem_savings))
        else:
            print('Estimated Building Demand Costs: ${:0.2f}'.format(est_build_demand_cost * (1 / frac_of_year)))
            print('Estimated Demand Cost After Storage Only: ${:0.2f}'.format(dem_cost_after_all * (1 / frac_of_year)))
            print('Estimated Demand Savings per year: ${:0.2f}'.format(demand_savings_total * (1 / frac_of_year)))
            print("Percent Demand Savings: {:0.2f}% \n".format(percent_dem_savings))

    general_building_cost = est_build_energy_cost + est_build_demand_cost
    general_cost_after_all = eng_cost_after_all + dem_cost_after_all
    gen_cost_after_solar = eng_cost_after_solar + dem_cost_after_solar
    gen_cost_storage = general_cost_after_all - gen_cost_after_solar

    general_savings_total = energy_savings_total + demand_savings_total
    general_savings_solar = energy_savings_solar + demand_savings_solar
    general_savings_storage = energy_savings_storage + demand_savings_storage

    percent_gen_saving = (general_savings_total / general_building_cost) * 100
    percent_storage_savings = (general_savings_storage / general_building_cost) * 100
    percent_solar_savings = (general_savings_solar / general_building_cost) * 100

    if verbose:
        if project_type == 'solar+storage':
            print("Estimated Building Total Costs: ${:0.2f}".format(general_building_cost * (1 / frac_of_year)))
            print('Estimated Total Cost After Solar+Storage: ${:0.2f}'.format(
                general_cost_after_all * (1 / frac_of_year)))
            print('Estimated Total Savings per year: ${:0.2f}'.format(general_savings_total * (1 / frac_of_year)),
                  'Solar portion: ${:0.2f}'.format(general_savings_solar * (1 / frac_of_year)),
                  'Storage portion: ${:0.2f}'.format(general_savings_storage * (1 / frac_of_year)))
            print("Percent Total Savings: {:0.2f}%".format(percent_gen_saving),
                  'Solar portion: {:0.2f}%'.format(percent_solar_savings),
                  'Storage Portion: {:0.2f}%'.format(percent_storage_savings))
        else:
            print("Estimated Building Total Costs: ${:0.2f}".format(general_building_cost * (1 / frac_of_year)))
            print(
                'Estimated Total Cost After Storage Only: ${:0.2f}'.format(general_cost_after_all * (1 / frac_of_year)))
            print('Estimated Total Savings per year: ${:0.2f}'.format(general_savings_total * (1 / frac_of_year)))
            print("Percent Total Savings: {:0.2f}%".format(percent_gen_saving))

    return general_building_cost, general_cost_after_all, gen_cost_after_solar


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def get_demand_cost(df):
    # category_demand_rate = []
    category_original = []
    category_total_solar = []
    category_total_solarandstorage = []

    for cat in set(df.demand_rate_category):
        category_demand_rate = (df.loc[df.demand_rate_category == cat].demand_rate)
        category_original_power = (df.loc[df.demand_rate_category == cat].original_building_power_kw)
        category_after_solar = (df.loc[df.demand_rate_category == cat].power_post_solar_kw)
        category_after_solarandstorage = (df.loc[df.demand_rate_category == cat].power_post_solarandstorage_kw)

        category_original.append(max(category_demand_rate * category_original_power))
        category_total_solar.append(max(category_demand_rate * category_after_solar))
        category_total_solarandstorage.append(max(category_demand_rate * category_after_solarandstorage))

    demand_cost_original = sum(category_original)
    demand_cost_solar = sum(category_total_solar)
    demand_cost_solarandstorage = sum(category_total_solarandstorage)
    return demand_cost_original, demand_cost_solar, demand_cost_solarandstorage


def generate_replicated_months(mnth, hourly_intervals, freq):
    """
    extrapolating a month's worth of data from a small sample within the particular month
    :param mnth: df of month to be extrapolated
    :param hourly_intervals: how many data points are within one hour
    :param freq: string representing the difference in time between each datapoint ('15min')
    :return: extrapolated month
    """
    num_days = monthrange(mnth.date.dt.year[1], mnth.date.dt.month[1])
    num_intervals = num_days[1] * 24 * hourly_intervals
    num_repl = 10

    month_repeat = pd.concat([mnth] * num_repl, ignore_index=True)  # estimation
    month_cut = []
    if len(month_repeat) > num_intervals:
        if mnth.date.dt.day[0] > 15:
            month_repeat = month_repeat.loc[month_repeat.index[-(num_intervals + 1):-1]]
        else:
            month_repeat = month_repeat.loc[month_repeat.index[0:num_intervals]]

    # TODO
    # freq intervals are related to hourly intervals (15m = 4 hourly intervals)
    # need to make an if segment for this

    datelist = pd.date_range((datetime.datetime(mnth.date.dt.year[0], mnth.date.dt.month[0], 1, 0, 0, 0)),
                             periods=num_intervals, freq=freq).tolist()
    # month_dates = month_dates.reset_index(drop=True)

    # for idx in range(len(month_dates)):
    #   month_dates[idx] = month_dates[idx].replace(year=2019)
    month_repeat['date'] = datelist

    return month_repeat.reset_index(drop=True)


def generate_replicated_year(df, hourly_intervals, freq, leap_year=False):
    """
    same as replicated_month, except extrapolating a year's data
    :param df: df of year to be extrapolated
    :param hourly_intervals: how many data points are within one hour
    :param freq: string representing the difference in time between each datapoint ('15min')
    :param leap_year: leap year or not
    :return: extrapolated year
    """
    if leap_year:
        num_days = 366
    else:
        num_days = 365
    num_intervals = num_days * 24 * hourly_intervals
    num_repl = 50

    year_repeat = pd.concat([df] * num_repl, ignore_index=True)  # estimation
    # month_cut = []
    if len(year_repeat) > num_intervals:
        if df.date.dt.month[0] > 6:
            year_repeat = year_repeat.loc[year_repeat.index[-(num_intervals + 1):-1]]
        else:
            year_repeat = year_repeat.loc[year_repeat.index[0:num_intervals]]

    # TODO
    # freq intervals are related to hourly intervals (15m = 4 hourly intervals)

    datelist = pd.date_range((datetime.datetime(df.date.dt.year[0], 1, 1, 0, 0, 0)), periods=num_intervals,
                             freq=freq).tolist()

    year_repeat['date'] = datelist

    return year_repeat.reset_index(drop=True)


def bargraph_ready(df, solar, hourly_intervals, freq, tariff_id, power, capacity, eff, itc, project_type, export):
    """
    create dictionaries ready for bargraph creation
    :param df: dataframe must contain 'date' and 'original_building_power_kw'
    :param solar: solar dataframe
    :param hourly_intervals: how many data points in an hour
    :param freq: string containing the time between each data point ('15min')
    :param tariff_id: demand/energy tariff id
    :param power: power of system
    :param capacity: capacity of system
    :param eff: efficiency of system
    :param itc: itc incentive eligibility
    :param project_type: solar+storage or storage only
    :param export: able to export power to grid
    :return original_bld_cost: dict with months as keys, original building cost as values
    :return only_solar_cost: dict with months as keys, post solar cost as values
    :return solar_and_storage_cost: dict with months as keys, post solar and storage cost as values
    """
    # data_needed = copy.deepcopy(df)

    original_bld_cost = dict.fromkeys(list(range(1, 12)))
    only_solar_cost = dict.fromkeys(list(range(1, 12)))
    solar_and_storage_cost = dict.fromkeys(list(range(1, 12)))

    for month_num in set(df.date.dt.month):
        month = df.loc[df.date.dt.month == month_num]
        # month = month.drop(columns='watts')
        month = month.reset_index(drop=True)
        month_repl = generate_replicated_months(month, hourly_intervals, freq)
        month_peak_shave = use_peak_shave_model(month_repl, solar, hourly_intervals=hourly_intervals,
                                                tariff_id=tariff_id, power=power, capacity=capacity,
                                                eff=eff, itc=itc, project_type=project_type, export=export)

        # suppress_stdout()
        with suppress_stdout():
            month_bld_cost, month_storageandsolar, month_just_solar = annual_savings(month_peak_shave)

        original_bld_cost[month_num] = month_bld_cost
        only_solar_cost[month_num] = month_just_solar
        solar_and_storage_cost[month_num] = month_storageandsolar

    for key in original_bld_cost:
        if original_bld_cost[key] == None:
            original_bld_cost[key] = 0

    for key in only_solar_cost:
        if only_solar_cost[key] == None:
            only_solar_cost[key] = 0

    for key in solar_and_storage_cost:
        if solar_and_storage_cost[key] == None:
            solar_and_storage_cost[key] = 0

    return original_bld_cost, only_solar_cost, solar_and_storage_cost


def superimposed_bargraphs(orig, sol, both, project_type):
    """
    generates a bargraph from dictionary (designed for solar+storage peak shaving)
    :param orig: dict1
    :param sol: dict2
    :param both: dict3
    :param project_type: solar+storage or storage only
    """

    idx = list(orig.keys())
    orig_val = list(orig.values())
    sol_val = list(sol.values())
    both_val = list(both.values())
    width = 0.8

    fig = plt.figure()

    plt.bar(idx, orig_val, width=width,
            color='blue', label='Original Building Cost')
    if project_type == 'solar+storage':
        plt.bar(idx, sol_val,
                width=width, color='yellow', alpha=1, label='Cost with Solar only')
        plt.bar(idx, both_val,
                width=width, color='green', alpha=1, label='Cost with both Solar and Storage')
    else:
        plt.bar(idx, both_val,
                width=width, color='green', alpha=1, label='Cost with Storage Only')

    ax = plt.axes()
    ax.get_yaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda y, p: format(int(y), ',')))
    ax.get_yaxis().set_major_formatter(
        matplotlib.ticker.FormatStrFormatter('$%d'))
    plt.xticks(idx)
    ax.grid(axis='y', alpha=.3, fillstyle='full')

    plt.ylabel('Cost')
    plt.xlabel('Months')
    if project_type == 'solar+storage':
        plt.title('Costs before and after Solar and Storage')
    else:
        plt.title('Costs before and after Storage')
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    fig.savefig('cost_before_and_after', bbox_extra_artists=(lgd,), bbox_inches='tight')


def peak_day_plot(full_df, start, finish, hourly_intervals, peak_month=4, peak_day=1, project_type='solar+storage'):
    """
    selects a day and outputs battery activity and power consumption on a graph
    :param full_df: dataframe containing all necessary columns (output from use_peak_shave_model func)
    :param start: start of the on peak time
    :param finish: end of the on peak time
    :param peak_month: choosing the month to use
    :param peak_day: choosing the day to use
    :param project_type: solar+storage or storage_only
    """

    peak_day = full_df.loc[(full_df.date.dt.month == peak_month) & (full_df.date.dt.day == peak_day)]
    peak_day = peak_day.reset_index(drop=True)
    peak_idx = peak_day[::hourly_intervals]
    peak_idx = peak_idx.index.tolist()
    fig = plt.figure(figsize=(15,5))
    plt.plot(peak_day.original_building_power_kw, 'blue', label='Original Building Power Consumption')
    if project_type == 'solar+storage':
        plt.plot(peak_day.power_post_solar_kw, 'yellow', label='Power Consumption with Solar only')
        plt.plot(peak_day.power_post_solarandstorage_kw, 'green', label='Power Consumption with Solar and Storage')
    else:
        plt.plot(peak_day.power_post_solarandstorage_kw, 'green', label='Power Consumption with Storage Only')

    plt.xticks(peak_idx, list(range(0, 24)))
    plt.xlabel('Hours')
    plt.ylabel('Power')
    plt.title('Peak day power consumption')
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    ax = plt.axes()
    ax.get_yaxis().set_major_formatter(
        matplotlib.ticker.FormatStrFormatter('%dkW'))
    ax.grid(axis='both', alpha=.3, color='gray')
    plt.axvspan(start * hourly_intervals, finish * hourly_intervals, alpha=0.2, color='gray')
    fig.savefig('power_before_and_after', bbox_extra_artists=(lgd,), bbox_inches='tight')


def demand_charge(df, power_type='original_building_power_kw'):
    """
    :param df: dataframe with demand_rate_categories as a column
    note: demand rate category columns need to be in the form demand_rate_category[number]
    :param power_type: which power column to use
    :return demand_charge: total demand charge for the dataframe
    """
    temp = []
    for col in df.columns:
        if 'demand_rate_category' in col:
            cat_num = 'demand_rate' + str(col[-1])
            for category in set(df[col]):
                max_point = df.loc[df[power_type] == max(df.loc[df[col] == category][power_type])]
                dem_chg = max_point[cat_num].iloc[0] * max_point[power_type].iloc[0]
                temp.append(dem_chg)

    demand_charge = sum(temp)
    return demand_charge

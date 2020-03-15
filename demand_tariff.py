import copy


def demand_tariff_transform(dataframe, tariff_id, date='date', power_data="original_building_power_kw"):
    dataframe = copy.deepcopy(dataframe)
    dic = {}
    demand_tariff_type = "monthly_peak_tod"
    if tariff_id == "ICE_Demand_Costa_Rica_T_MT":
        # create demand rate column
        dataframe["demand_rate"] = [0] * len(dataframe.index)

        # peak criteria:
        peak_criteria1 = (((dataframe[date].dt.weekday >= 0) & (dataframe[date].dt.weekday <= 4)) &
                          ((dataframe[date].dt.hour > 9) & (dataframe[date].dt.hour < 13))) | \
                         (((dataframe[date].dt.weekday >= 0) & (dataframe[date].dt.weekday <= 4)) &
                          ((dataframe[date].dt.hour > 16) & (dataframe[date].dt.hour < 20)))

        valley_criteria2 = ((dataframe[date].dt.hour >= 6) & (dataframe[date].dt.hour <= 9)) | \
                           ((dataframe[date].dt.hour >= 13) & (dataframe[date].dt.hour <= 16))

        night_criteria3 = ((dataframe[date].dt.hour >= 20) & (dataframe[date].dt.hour <= 23)) | \
                          ((dataframe[date].dt.hour >= 0) & (dataframe[date].dt.hour <= 5))

        # assigning rates
        dataframe.loc[peak_criteria1, "demand_rate"] = 18.48
        dataframe.loc[valley_criteria2, "demand_rate"] = 12.90
        dataframe.loc[night_criteria3, "demand_rate"] = 8.27

        # finding the cost for each data point
        dataframe["demand_cost_load_usd"] = dataframe[power_data] * dataframe["demand_rate"]

        # assigning categories
        dataframe["demand_rate_category"] = [0] * len(dataframe.index)
        dataframe.loc[dataframe["demand_rate"] == 18.48, "demand_rate_category"] = "peak_period"
        dataframe.loc[dataframe["demand_rate"] == 12.90, "demand_rate_category"] = "valley_period"
        dataframe.loc[dataframe["demand_rate"] == 8.27, "demand_rate_category"] = "night_period"
        demand_tariff_type = "monthly_peak_tod"

        dic["dataframe"] = dataframe
        dic["list"] = dataframe[["date", "demand_cost_load_usd"]]

    elif tariff_id == "ConEd_SC9_Demand_Rate5_2019":
        # Multiple dwelling service, with front of the meter storage
        # cdc : contract_demand_charges
        cdc = 6.67 # applicable for all months
        print('subtract {} per kw per month from your savings '.format(cdc))
        #  June, July, August, and September Monday through Friday, 8 AM to 6 PM $0.5150 per kW
        # Monday through Friday, 8 AM to 10 PM $1.3268 per kW
        # ALl other months Monday through Friday, 8 AM to 10 PM $0.9604 per kW

        # creating demand rate/ demand cost columns
        dataframe["demand_cost_load_usd"] = [0] * len(dataframe.index)
        dataframe["demand_rate_category"] = 'no_charges'
        dataframe['demand_rate'] = 0

        dataframe["peak_demand_cost_type"] = "daily_peak_step"

        peak = dataframe[date].dt.month.isin([6,7,8,9])
        off_peak = dataframe[date].dt.month.isin([1,2,3,4,5,10,11,12])
        week_days = dataframe[date].dt.dayofweek.isin([0,1,2,3,4])
        peak_hours1 = dataframe[date].dt.hour.isin([18,19,20,21])
        peak_hours2 = dataframe[date].dt.hour.isin([8,9,10,11,12,13,14,15,16,17])
        off_peak_hours = dataframe[date].dt.hour.isin([8,9,10,11,12,13,14,15,16,17,18,19,20,21])

        dataframe.loc[peak & week_days & peak_hours1, 'demand_rate'] = 0.3420
        dataframe.loc[peak & week_days & peak_hours1, 'demand_rate_category'] = 'peak_low'

        dataframe.loc[peak & week_days & peak_hours2, 'demand_rate'] = 0.3420 + 0.5002
        dataframe.loc[peak & week_days & peak_hours2, 'demand_rate_category'] = 'peak_high'

        dataframe.loc[off_peak & week_days & off_peak_hours, 'demand_rate'] = 0.4560
        dataframe.loc[off_peak & week_days & off_peak_hours, 'demand_rate_category'] = 'off_peak'

        dataframe['demand_cost_load_usd'] = 0

        dic["dataframe"] = dataframe
        dic["list"] = dataframe[["demand_rate_category", "demand_cost_load_usd", date]]

    elif tariff_id == "ConEd_SC12_Demand_Rate4_2019":
        # Multiple dwelling service, with front of the meter storage
        # cdc : contract_demand_charges
        cdc = 7.18 # applicable for all months
        print('subtract {} per kw per month from your savings '.format(cdc))
        #  June, July, August, and September Monday through Friday, 8 AM to 6 PM $0.5150 per kW
        # Monday through Friday, 8 AM to 10 PM $1.3268 per kW
        # ALl other months Monday through Friday, 8 AM to 10 PM $0.9604 per kW

        # creating demand rate/ demand cost columns
        dataframe["demand_cost_load_usd"] = [0] * len(dataframe.index)
        dataframe["demand_rate_category"] = 'no_charges'
        dataframe['demand_rate'] = 0

        dataframe["peak_demand_cost_type"] = "daily_peak_step"

        peak = dataframe[date].dt.month.isin([6,7,8,9])
        off_peak = dataframe[date].dt.month.isin([1,2,3,4,5,10,11,12])
        week_days = dataframe[date].dt.dayofweek.isin([0,1,2,3,4])
        peak_hours1 = dataframe[date].dt.hour.isin([18,19,20,21])
        peak_hours2 = dataframe[date].dt.hour.isin([8,9,10,11,12,13,14,15,16,17])
        off_peak_hours = dataframe[date].dt.hour.isin([8,9,10,11,12,13,14,15,16,17,18,19,20,21])

        dataframe.loc[peak & week_days & peak_hours1, 'demand_rate'] = 1.3268
        dataframe.loc[peak & week_days & peak_hours1, 'demand_rate_category'] = 'peak_high'

        dataframe.loc[peak & week_days & peak_hours2, 'demand_rate'] = 1.3268 + 0.5150
        dataframe.loc[peak & week_days & peak_hours2, 'demand_rate_category'] = 'peak_low'

        dataframe.loc[off_peak & week_days & off_peak_hours, 'demand_rate'] = 0.9604
        dataframe.loc[off_peak & week_days & off_peak_hours, 'demand_rate_category'] = 'off_peak'

        dataframe['demand_cost_load_usd'] = 0

        dic["dataframe"] = dataframe
        dic["list"] = dataframe[["demand_rate_category", "demand_cost_load_usd", date]]

    elif tariff_id == "ConEd_SC8_Demand_Rate1_2019":
        # creating month column
        if 'month' not in dataframe.columns:
            dataframe['month'] = dataframe.date.dt.month

        # creating demand rate/ demand cost columns
        dataframe["demand_cost_load_usd"] = [0] * len(dataframe.index)
        dataframe["demand_rate_category"] = [0] * len(dataframe.index)
        dataframe["peak_demand_cost_type"] = "monthly_peak_step"

        # conditions
        peak = ((dataframe["month"] >= 6) & (dataframe["month"] <= 9))
        off_peak = ((dataframe["month"] <= 5) | (dataframe["month"] >= 10))

        # assigning rates based on conditions
        tax = .045
        dataframe["demand_rate"] = 0
        dataframe.loc[peak, "demand_rate"] = 33.9 * (1+tax)
        dataframe.loc[off_peak, "demand_rate"] = 26.19 * (1+tax)
        dataframe['demand_cost_load_usd'] = dataframe[power_data] * dataframe['demand_rate']
        # assigning categories
        dataframe.loc[(dataframe["month"] >= 6) & (dataframe["month"] <= 9), "demand_rate_category"] = "peak"
        dataframe.loc[(dataframe["month"] <= 5) | (dataframe["month"] >= 10), "demand_rate_category"] = "off_peak"

        dic["dataframe"] = dataframe
        dic["list"] = dataframe[["demand_rate_category", "demand_cost_load_usd", date]]

    elif tariff_id == "ConEd_SC8_Rate2_2019":
        if 'month' not in dataframe.columns:
            dataframe['month'] = dataframe[date].dt.month
        dataframe["demand_cost_load_usd"] = [0] * len(dataframe.index)
        dataframe["demand_rate1"] = [0] * len(dataframe.index)
        dataframe["demand_rate2"] = [0] * len(dataframe.index)
        dataframe["demand_rate3"] = [0] * len(dataframe.index)
        dataframe["demand_rate_category1"] = 'no_charge'
        dataframe["demand_rate_category2"] = 'no_charge'
        dataframe["demand_rate_category3"] = 'no_charge'
        dataframe["peak_demand_cost_type"] = "monthly_peak_tod"

        # conditions
        """
        Summer Time of Use Level1 ($/kW): 9.06 (weekdays, 8AM to 6PM)
        Summer Time of Use Level2 ($/kW): 21.84 (weekdays, 8AM to 10PM)
        Summer Time of Use Level3 ($/kW): 17.74 (all days, all hours)
        Winter Time of Use Level1 ($/kW): 15.98 (weekdays, 8AM to 10PM)
        Winter Time of Use Level2 ($/kW): 3.74 (all days, all hours)

        """

        summer_tou_1 = (((dataframe[date].dt.weekday >= 0) & (dataframe[date].dt.weekday <= 4)
                         ) & ((dataframe[date].dt.hour >= 8) & (dataframe[date].dt.hour <= 17))
                        & ((dataframe[date].dt.month >= 6) & (dataframe[date].dt.month <= 9)))
        summer_tou_2 = (((dataframe[date].dt.weekday >= 0) & (dataframe[date].dt.weekday <= 4)
                         ) & ((dataframe[date].dt.hour >= 8) & (dataframe[date].dt.hour <= 21))
                        & ((dataframe[date].dt.month >= 6) & (dataframe[date].dt.month <= 9)))
        summer_tou_3 = ((dataframe[date].dt.month >= 6) & (dataframe[date].dt.month <= 9))

        winter_tou_1 = (((dataframe[date].dt.weekday >= 0) & (dataframe[date].dt.weekday <= 4)
                         ) & ((dataframe[date].dt.hour >= 8) & (dataframe[date].dt.hour <= 21))
                        & ((dataframe[date].dt.month <= 5) | (dataframe[date].dt.month >= 10)))
        winter_tou_2 = ((dataframe[date].dt.month <= 5) | (dataframe[date].dt.month >= 10))

        # assigning rates based on conditions
        dataframe.loc[summer_tou_1, 'demand_rate1'] = 9.06
        dataframe.loc[summer_tou_2, 'demand_rate2'] = 21.84
        dataframe.loc[summer_tou_3, 'demand_rate3'] = 17.74
        dataframe.loc[winter_tou_1, 'demand_rate1'] = 15.98
        dataframe.loc[winter_tou_2, 'demand_rate2'] = 3.74

        # finding cost for all data points
        dataframe["demand_cost_load_usd"] = (dataframe["demand_rate1"]+dataframe["demand_rate2"] +
                                             dataframe["demand_rate3"]) * dataframe[power_data]

        # assigning categories
        dataframe.loc[summer_tou_1, 'demand_rate_category1'] = "Summer TOU Peak"
        dataframe.loc[summer_tou_2, 'demand_rate_category2'] = "Summer TOU Off-Peak"
        dataframe.loc[summer_tou_3, 'demand_rate_category3'] = "Summer TOU Super Off-Peak"

        dataframe.loc[winter_tou_1, 'demand_rate_category1'] = "Winter TOU Off-Peak"
        dataframe.loc[winter_tou_2, 'demand_rate_category2'] = "Winter TOU Super Off-Peak"

        # add to dictionary
        dic["dataframe"] = dataframe
        dic["list"] = dataframe[["demand_rate_category1", "demand_cost_load_usd"]]

    elif tariff_id == "ConEd_SC8_Demand_Rate4_2019":
        # Multiple dwelling service, with front of the meter storage
        # cdc : contract_demand_charges
        cdc = 7.73  # applicable for all months
        print('subtract {} per kw per month from your savings '.format(cdc))
        #  June, July, August, and September Monday through Friday, 8 AM to 6 PM $0.5150 per kW
        # Monday through Friday, 8 AM to 10 PM $1.3268 per kW
        # ALl other months Monday through Friday, 8 AM to 10 PM $0.9604 per kW

        # creating demand rate/ demand cost columns
        dataframe["demand_cost_load_usd"] = [0] * len(dataframe.index)
        dataframe["demand_rate_category"] = 'no_charges'
        dataframe['demand_rate'] = 0

        dataframe["peak_demand_cost_type"] = "daily_peak_step"

        peak = dataframe[date].dt.month.isin([6, 7, 8, 9])
        off_peak = dataframe[date].dt.month.isin([1, 2, 3, 4, 5, 10, 11, 12])
        week_days = dataframe[date].dt.dayofweek.isin([0, 1, 2, 3, 4])
        peak_hours1 = dataframe[date].dt.hour.isin([18, 19, 20, 21])
        peak_hours2 = dataframe[date].dt.hour.isin([8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
        off_peak_hours = dataframe[date].dt.hour.isin([8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])

        dataframe.loc[peak & week_days & peak_hours1, 'demand_rate'] = 1.4917
        dataframe.loc[peak & week_days & peak_hours1, 'demand_rate_category'] = 'peak_high'

        dataframe.loc[peak & week_days & peak_hours2, 'demand_rate'] = 1.4917 + 0.7020
        dataframe.loc[peak & week_days & peak_hours2, 'demand_rate_category'] = 'peak_low'

        dataframe.loc[off_peak & week_days & off_peak_hours, 'demand_rate'] = 0.9771
        dataframe.loc[off_peak & week_days & off_peak_hours, 'demand_rate_category'] = 'off_peak'

        dataframe['demand_cost_load_usd'] = 0

        dic["dataframe"] = dataframe
        dic["list"] = dataframe[["demand_rate_category", "demand_cost_load_usd", date]]

    elif tariff_id == "SDGE_AL_TOU_Industrial_2018":
        if 'month' not in dataframe.columns:
            dataframe['month'] = dataframe[date].dt.month
        dataframe["demand_cost_load_usd"] = [0] * len(dataframe.index)
        dataframe["demand_rate_category"] = [0] * len(dataframe.index)
        dataframe["demand_rate"] = [0] * len(dataframe.index)
        demand_tariff_type = "monthly_peak_tod"

        # conditions
        summer_on_peak = ((dataframe["month"] >= 6) | (dataframe["month"] <= 10))
        winter_on_peak = ((dataframe["month"] >= 11) | (dataframe["month"] <= 5))

        # assigning rates based on conditions
        dataframe.loc[summer_on_peak, "demand_rate"] = 16.48 + 21
        dataframe.loc[winter_on_peak, "demand_rate"] = 16.44 + 21
        # finding cost for all data points
        dataframe["demand_cost_load_usd"] = dataframe["demand_rate"] * dataframe[power_data]
        # assigning categories
        dataframe.loc[(dataframe["month"] >= 6) & (dataframe["month"] <= 10), "demand_rate_category"] = "summer"
        dataframe.loc[(dataframe["month"] <= 5) | (dataframe["month"] >= 11), "demand_rate_category"] = "winter"
        # add to dictionary
        dic["dataframe"] = dataframe
        dic["list"] = dataframe[["demand_rate_category", "demand_cost_load_usd"]]

    elif tariff_id == "SDGE_AL_TOU_Industrial_2019":
        if 'month' not in dataframe.columns:
            dataframe['month'] = dataframe[date].dt.month
        dataframe["demand_cost_load_usd"] = [0] * len(dataframe.index)
        dataframe["demand_rate_category"] = [0] * len(dataframe.index)
        dataframe["demand_rate"] = [0] * len(dataframe.index)
        demand_tariff_type = "monthly_peak_tod"

        # conditions
        summer_on_peak = ((dataframe["month"] >= 6) | (dataframe["month"] <= 10))
        winter_on_peak = ((dataframe["month"] >= 11) | (dataframe["month"] <= 5))

        # assigning rates based on conditions
        dataframe.loc[summer_on_peak, "demand_rate"] = 16.96 + 21.34
        dataframe.loc[winter_on_peak, "demand_rate"] = 16.98 + 21.34
        # finding cost for all data points
        dataframe["demand_cost_load_usd"] = dataframe["demand_rate"] * dataframe[power_data]
        # assigning categories
        dataframe.loc[(dataframe["month"] >= 6) & (dataframe["month"] <= 10), "demand_rate_category"] = "summer"
        dataframe.loc[(dataframe["month"] <= 5) | (dataframe["month"] >= 11), "demand_rate_category"] = "winter"
        # add to dictionary
        dic["dataframe"] = dataframe
        dic["list"] = dataframe[["demand_rate_category", "demand_cost_load_usd", date]]

    elif tariff_id == "PSEGLI_285_2019":
        # creating columns'
        dataframe["demand_cost_load_usd"] = [0] * len(dataframe.index)
        dataframe["demand_rate_category"] = [0] * len(dataframe.index)
        demand_tariff_type = "monthly_peak_tod"
        # conditions
        off_peak = ((dataframe[date].dt.hour >= 0) & (dataframe[date].dt.hour < 7))

        on_peak = ((dataframe[date].dt.month >= 6) & (dataframe[date].dt.month <= 9)) & (
                (dataframe[date].dt.weekday >= 0) &
                (dataframe[date].dt.weekday <= 5)) & \
                  ((dataframe[date].dt.hour >= 10) & (dataframe[date].dt.hour < 22))

        # intermediate Demand_Rate is set for the entire column,
        # and other Demand_Rates overwrite it when the conditions are met
        dataframe["demand_rate"] = 6.43
        dataframe.loc[off_peak, "demand_rate"] = 0
        dataframe.loc[on_peak, "demand_rate"] = 26.97
        # finding cost for all data points, then assigning categories
        dataframe["demand_cost_load_usd"] = dataframe[power_data] * dataframe["demand_rate"]
        dataframe.loc[dataframe["demand_rate"] == 6.43, "demand_rate_category"] = "intermediate"
        dataframe.loc[dataframe["demand_rate"] == 0, "demand_rate_category"] = "off_peak"
        dataframe.loc[dataframe["demand_rate"] == 26.97, "demand_rate_category"] = "on_peak"
        # add to dictionary
        dic["dataframe"] = dataframe
        dic["list"] = dataframe[[date, "demand_cost_load_usd"]]

    elif tariff_id == "NvEnergy_LGS2_S":
        """
        Summer (June – September)
        On-Peak 1:01 p.m. – 7:00 p.m. Daily (T)
        Mid-Peak 10:01 a.m. - 1:00 p.m. and 7:01 p.m. – 10:00 p.m. Daily (T)
        Off-Peak 10:01 p.m. – 10:00 a.m. Daily (T)
        Winter All Other (October - May)
        """
        df_date = dataframe.set_index(date, drop=False)
        on_peak_idx = df_date.between_time('13:01', '19:00')[date].dt.month.isin([6, 7, 8, 9])
        on_peak_idx = on_peak_idx[on_peak_idx.values].index
        mid_peak_idx1 = df_date.between_time('10:01', '13:00')[date].dt.month.isin([6, 7, 8, 9])
        mid_peak_idx1 = mid_peak_idx1[mid_peak_idx1.values].index
        mid_peak_idx2 = df_date.between_time('19:01', '22:00')[date].dt.month.isin([6, 7, 8, 9])
        mid_peak_idx2 = mid_peak_idx2[mid_peak_idx2.values].index
        off_peak_idx = df_date.between_time('22:01', '10:00')[date].dt.month.isin([6, 7, 8, 9])
        off_peak_idx = off_peak_idx[off_peak_idx.values].index

        dataframe['demand_rate'] = 3.14 + 0.40
        dataframe.loc[dataframe[date].isin(on_peak_idx), 'demand_rate'] = 3.14 + 13.35
        dataframe.loc[dataframe[date].isin(mid_peak_idx1), 'demand_rate'] = 3.14 + 2.04
        dataframe.loc[dataframe[date].isin(mid_peak_idx2), 'demand_rate'] = 3.14 + 2.04
        dataframe.loc[dataframe[date].isin(off_peak_idx), 'demand_rate'] = 3.14 + 0

        dataframe.loc[dataframe.demand_rate == 3.14 + 0.40, "demand_rate_category"] = "winter_off_peak"
        dataframe.loc[dataframe.demand_rate == 3.14 + 13.35, "demand_rate_category"] = "summer_on_peak"
        dataframe.loc[dataframe.demand_rate == 3.14 + 0, "demand_rate_category"] = "summer_off_peak"
        dataframe.loc[dataframe.demand_rate == 3.14 + 2.04, "demand_rate_category"] = "summer_mid_peak"

        # add type and cost load
        dataframe["demand_cost_load_usd"] = dataframe[power_data] * dataframe["demand_rate"]
        demand_tariff_type = "monthly_peak_tod"
        # add to dictionary
        dic["dataframe"] = dataframe
        dic["list"] = dataframe[['demand_rate_category', "demand_cost_load_usd"]]

    elif tariff_id == "NvEnergy_LGS3_S":
        """
        Summer (June – September)
        On-Peak 1:01 p.m. – 7:00 p.m. Daily (T)
        Mid-Peak 10:01 a.m. - 1:00 p.m. and 7:01 p.m. – 10:00 p.m. Daily (T)
        Off-Peak 10:01 p.m. – 10:00 a.m. Daily (T)
        Winter All Other (October - May)
        """
        df_date = dataframe.set_index(date, drop=False)
        on_peak_idx = df_date.between_time('13:01', '19:00')[date].dt.month.isin([6, 7, 8, 9])
        on_peak_idx = on_peak_idx[on_peak_idx.values].index
        mid_peak_idx1 = df_date.between_time('10:01', '13:00')[date].dt.month.isin([6, 7, 8, 9])
        mid_peak_idx1 = mid_peak_idx1[mid_peak_idx1.values].index
        mid_peak_idx2 = df_date.between_time('19:01', '22:00')[date].dt.month.isin([6, 7, 8, 9])
        mid_peak_idx2 = mid_peak_idx2[mid_peak_idx2.values].index
        off_peak_idx = df_date.between_time('22:01', '10:00')[date].dt.month.isin([6, 7, 8, 9])
        off_peak_idx = off_peak_idx[off_peak_idx.values].index

        dataframe['demand_rate'] = 3.38 + 0.55
        dataframe.loc[dataframe[date].isin(on_peak_idx), 'demand_rate'] = 3.38 + 16.30
        dataframe.loc[dataframe[date].isin(mid_peak_idx1), 'demand_rate'] = 3.38 + 2.59
        dataframe.loc[dataframe[date].isin(mid_peak_idx2), 'demand_rate'] = 3.38 + 2.59
        dataframe.loc[dataframe[date].isin(off_peak_idx), 'demand_rate'] = 3.38 + 0

        dataframe.loc[dataframe.demand_rate == 3.38 + 0.55, "demand_rate_category"] = "winter_off_peak"
        dataframe.loc[dataframe.demand_rate == 3.38 + 16.30, "demand_rate_category"] = "summer_on_peak"
        dataframe.loc[dataframe.demand_rate == 3.38 + 0, "demand_rate_category"] = "summer_off_peak"
        dataframe.loc[dataframe.demand_rate == 3.38 + 2.59, "demand_rate_category"] = "summer_mid_peak"

        # add type and cost load
        dataframe["demand_cost_load_usd"] = dataframe[power_data] * dataframe["demand_rate"]
        demand_tariff_type = "monthly_peak_tod"
        # add to dictionary
        dic["dataframe"] = dataframe
        dic["list"] = dataframe[['demand_rate_category', "demand_cost_load_usd"]]

    elif tariff_id == "RIT_SC8_Primary_RG&E_2019":
        # creating columns
        dataframe["demand_cost_load_usd"] = [0] * len(dataframe.index)
        dataframe["demand_rate_category"] = [0] * len(dataframe.index)
        demand_tariff_type = "monthly_peak_tod"
        # conditions: flat demand charge of 15
        dataframe["demand_rate"] = 15

        # finding cost for all data points, then assigning categories
        dataframe["demand_cost_load_usd"] = dataframe[power_data] * dataframe["demand_rate"]
        dataframe.loc[dataframe["demand_rate"] == 15, "demand_rate_category"] = "flat_demand_charge"

        # add to dictionary
        dic["dataframe"] = dataframe
        dic["list"] = dataframe[[date, "demand_cost_load_usd"]]

    elif tariff_id == 'eversource_greater_boston_b3_g6':
        """
        Peak is from 9 a.m. to 6 p.m. weekdays from June through September; and 8 a.m. to 9 p.m. 
        weekdays October through May. Off-peak is all other hours including weekends and Massachusetts holidays
        - peak demand rate = 
        - off peak demand rate =  
        """
        dataframe['demand_rate'] = 0
        dataframe['demand_rate_category'] = ''
        weekday_condition = dataframe[date].dt.weekday.isin([0, 1, 2, 3, 4])
        hours = [hour for hour in range(9,19)]
        hour_condition1 = dataframe[date].dt.hour.isin(hours)
        months = [6, 7, 8, 9]
        months_condition1 = dataframe[date].dt.month.isin(months)
        condition1 = weekday_condition & hour_condition1 & months_condition1
        # demand rate for condition1 = 15.04+9.05      8.87 + 9.05
        dataframe.loc[months_condition1, 'demand_rate'] = (15.04 + 9.05) * .7
        dataframe.loc[months_condition1, 'demand_rate_category'] = 'summer_on_peak'
        dataframe.loc[condition1, 'demand_rate'] = 15.04 + 9.05
        dataframe.loc[condition1, 'demand_rate_category'] = 'summer_on_peak'
        # demand rate for condition2 = 8.87 + 9.05
        hours = [hour for hour in range(8, 22)]
        hour_condition2 = dataframe[date].dt.hour.isin(hours)
        months = [10, 11, 12, 1, 2, 3, 4, 5]
        months_condition2 = dataframe[date].dt.month.isin(months)
        condition2 = weekday_condition & hour_condition2 & months_condition2
        dataframe.loc[months_condition2, 'demand_rate'] = (8.87 + 9.05) * .7
        dataframe.loc[months_condition2, 'demand_rate_category'] = 'winter_on_peak'
        dataframe.loc[condition2, 'demand_rate'] = 8.87 + 9.05
        dataframe.loc[condition2, 'demand_rate_category'] = 'winter_on_peak'

        # add cost load
        dataframe["demand_cost_load_usd"] = dataframe[power_data] * dataframe["demand_rate"]
        # add to dictionary
        dic["dataframe"] = dataframe
        dic["list"] = dataframe[['demand_rate_category', "demand_cost_load_usd", date]]
        
    elif tariff_id == 'flat_rate':
        dataframe['demand_rate'] = 9
        dataframe['demand_rate_category'] = 'flat_demand_rate'
        
         # add cost load
        dataframe["demand_cost_load_usd"] = dataframe[power_data] * dataframe["demand_rate"]
        # add to dictionary
        dic["dataframe"] = dataframe
        dic["list"] = dataframe[['demand_rate_category', "demand_cost_load_usd", date]]

        #TODO, change calculator to make this work - demand rate depends on power post solar kw
    elif tariff_id == 'ConEd_SC9_Demand_Rate1_2019':


        dataframe["demand_rate"] = [0] * len(dataframe.index)
        dataframe["demand_rate_category"] = [0] * len(dataframe.index)


        on_peak1 = ((dataframe['date'].dt.month >= 6) & (dataframe['date'].dt.month <= 9) & (dataframe['power_post_solar_kw'] > 900))
        on_peak2 = ((dataframe['date'].dt.month >= 6) & (dataframe['date'].dt.month <= 9) & (dataframe['power_post_solar_kw'] <= 900))
        off_peak1 = ((dataframe['date'].dt.month <=5) | (dataframe['date'].dt.month >= 10) & (dataframe['power_post_solar_kw'] > 900))
        off_peak2 = ((dataframe['date'].dt.month <=5) | (dataframe['date'].dt.month >= 10) & (dataframe['power_post_solar_kw'] <= 900))

        dataframe.loc[on_peak1, "demand_rate"] = ((900*(17.61-15.91))/dataframe['power_post_solar_kw'])+15.90 
        dataframe.loc[on_peak2, "demand_rate"] = 17.61
        dataframe.loc[off_peak1, "demand_rate"] = ((900*(14.07-12.35))/dataframe['power_post_solar_kw'])+12.35 
        dataframe.loc[off_peak2, "demand_rate"] = 14.07
            
        dataframe.loc[dataframe["demand_rate"] == ((900*(17.61-15.91))/dataframe['power_post_solar_kw'])+15.90, "demand_rate_category"] = "on_peak"
        dataframe.loc[dataframe["demand_rate"] == 17.61, "demand_rate_category"] = "on_peak"
        dataframe.loc[dataframe["demand_rate"] == ((900*(14.07-12.35))/dataframe['power_post_solar_kw'])+12.35, "demand_rate_category"] = "off_peak"
        dataframe.loc[dataframe["demand_rate"] == 14.07, "demand_rate_category"] = "off_peak"
        
         # add cost load
        dataframe["demand_cost_load_usd"] = dataframe[power_data] * dataframe["demand_rate"]
        # add to dictionary
        dic["dataframe"] = dataframe
        dic["list"] = dataframe[['demand_rate_category', "demand_cost_load_usd", date]]
        
    else:
        raise NameError('tariff_id is not found')
    return dic, demand_tariff_type

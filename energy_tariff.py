import copy


def energy_tariff_transform(dataframe, tariff_id, date='date', energy_data="original_building_energy_kwh"):
    """
    tariff_id can be one of those: ['PSEGLI_285_2019', 'SDGE_AL_TOU_Industrial_2018']
    :param dataframe:
    :param tariff_id:
    :param energy_data:
    :param date:
    :return:
    """
    dataframe = copy.deepcopy(dataframe)
    dic = {}
    if tariff_id == "PSEGLI_285_2019":
        dataframe["energy_cost_load_usd"] = [0] * len(dataframe.index)
        dataframe["energy_rate_category"] = [0] * len(dataframe.index)
        dataframe["energy_cost_type"] = "monthly_tod"

        off_peak = ((dataframe[date].dt.hour >= 0) & (dataframe[date].dt.hour < 7))
        on_peak = ((dataframe[date].dt.month >= 6) & (dataframe[date].dt.month <= 9)) & (
                (dataframe[date].dt.weekday >= 0) & (dataframe[date].dt.weekday <= 5)) & (
                          (dataframe[date].dt.hour >= 10) & (dataframe[date].dt.hour < 22))

        # intermediate Energy_Rate is set for the entire column,
        # and other Energy_Rates overwrite it when the conditions are met
        supply_charge = .11
        dataframe["energy_rate"] = supply_charge + 0.0228
        dataframe.loc[off_peak, "energy_rate"] = supply_charge + 0.0055
        dataframe.loc[on_peak, "energy_rate"] = supply_charge + 0.0357

        dataframe["energy_cost_load_usd"] = dataframe[energy_data] * dataframe["energy_rate"]
        dataframe.loc[dataframe["energy_rate"] == supply_charge + 0.0228, "energy_rate_category"] = "intermediate"
        dataframe.loc[dataframe["energy_rate"] == supply_charge + 0.0055, "energy_rate_category"] = "off_peak"
        dataframe.loc[dataframe["energy_rate"] == supply_charge + 0.0357, "energy_rate_category"] = "on_peak"

        dic["dataframe"] = dataframe
        dic["List"] = dataframe[[date, "energy_cost_load_usd"]]

    elif tariff_id == "SDGE_AL_TOU_Industrial_2018":
        dataframe["energy_cost_load_usd"] = [0] * len(dataframe.index)
        dataframe["energy_rate_category"] = [0] * len(dataframe.index)
        dataframe["energy_rate"] = [0] * len(dataframe.index)
        dataframe["energy_cost_type"] = "monthly_tod"

        on_peak = ((dataframe[date].dt.hour > 15) & (dataframe[date].dt.hour < 21)) & (
                (dataframe["month"] >= 11) | (dataframe["month"] <= 5))
        off_peak = (((dataframe[date].dt.hour > 5) & (dataframe[date].dt.hour < 16)) | (
                (dataframe[date].dt.hour > 20) & (dataframe[date].dt.hour <= 23))) & (
                           (dataframe["month"] >= 11) | (dataframe["month"] <= 5))

        # Super_Off_Peak is All other hours

        dataframe["energy_rate"] = 0.09
        dataframe.loc[on_peak, "energy_rate"] = 0.115
        dataframe.loc[off_peak, "energy_rate"] = .103
        dataframe["energy_cost_load_usd"] = dataframe[energy_data] * dataframe["energy_rate"]
        dataframe.loc[dataframe["energy_rate"] == 0.09, "energy_rate_category"] = "super_off_peak"
        dataframe.loc[dataframe["energy_rate"] == 0.115, "energy_rate_category"] = "on_peak"
        dataframe.loc[dataframe["energy_rate"] == 0.103, "energy_rate_category"] = "off_peak"

        dic["dataframe"] = dataframe
        dic["list"] = dataframe[["demand_rate_category", "demand_cost_load_usd"]]

    elif tariff_id == "SDGE_AL_TOU_Industrial_2019":
        dataframe["energy_cost_load_usd"] = [0] * len(dataframe.index)
        dataframe["energy_rate_category"] = [0] * len(dataframe.index)
        dataframe["energy_rate"] = [0] * len(dataframe.index)
        dataframe["energy_cost_type"] = "monthly_tod"

        on_peak = ((dataframe[date].dt.hour > 15) & (dataframe[date].dt.hour < 21)) & (
                (dataframe["month"] >= 11) | (dataframe["month"] <= 5))
        off_peak = (((dataframe[date].dt.hour > 5) & (dataframe[date].dt.hour < 16)) | (
                (dataframe[date].dt.hour > 20) & (dataframe[date].dt.hour <= 23))) & (
                           (dataframe["month"] >= 11) | (dataframe["month"] <= 5))

        # Super_Off_Peak is All other hours

        dataframe["energy_rate"] = 0.095
        dataframe.loc[on_peak, "energy_rate"] = 0.12
        dataframe.loc[off_peak, "energy_rate"] = .108
        dataframe["energy_cost_load_usd"] = dataframe[energy_data] * dataframe["energy_rate"]
        dataframe.loc[dataframe["energy_rate"] == 0.095, "energy_rate_category"] = "super_off_peak"
        dataframe.loc[dataframe["energy_rate"] == 0.12, "energy_rate_category"] = "on_peak"
        dataframe.loc[dataframe["energy_rate"] == 0.108, "energy_rate_category"] = "off_peak"

        dic["dataframe"] = dataframe
        dic["list"] = dataframe[["demand_rate_category", "demand_cost_load_usd"]]

    elif tariff_id == "NvEnergy_LGS2_S":
        # here energy_rate_category = demand_rate_category so no need to create duplicate columns
        dataframe["energy_rate"] = 0
        dataframe["energy_cost_type"] = "monthly_tod"
        dataframe["energy_rate_category"] = dataframe["demand_rate_category"]
        fixed = .00067 + .00077 + .00114 - .00238
        dataframe.loc[dataframe["demand_rate_category"] == "winter_off_peak", 'energy_rate'] = .05178 + fixed
        dataframe.loc[dataframe["demand_rate_category"] == "summer_on_peak", 'energy_rate'] = .08473 + fixed
        dataframe.loc[dataframe["demand_rate_category"] == "summer_off_peak", 'energy_rate'] = .04538 + fixed
        dataframe.loc[dataframe["demand_rate_category"] == "summer_mid_peak", 'energy_rate'] = .06414 + fixed
        dataframe["energy_cost_load_usd"] = dataframe[energy_data] * dataframe["energy_rate"]
        dic["dataframe"] = dataframe
        dic["list"] = dataframe[["demand_rate_category", "demand_cost_load_usd"]]

    elif tariff_id == "NvEnergy_LGS3_S":
        dataframe["energy_rate"] = 0
        dataframe["energy_cost_type"] = "monthly_tod"
        dataframe["energy_rate_category"] = dataframe["demand_rate_category"]
        fixed = .00067 + .00077 + 0.00105 - .00224
        dataframe.loc[dataframe["demand_rate_category"] == "winter_off_peak", 'energy_rate'] = 0.04934 + fixed
        dataframe.loc[dataframe["demand_rate_category"] == "summer_on_peak", 'energy_rate'] = 0.07981 + fixed
        dataframe.loc[dataframe["demand_rate_category"] == "summer_off_peak", 'energy_rate'] = 0.04330 + fixed
        dataframe.loc[dataframe["demand_rate_category"] == "summer_mid_peak", 'energy_rate'] = 0.06122 + fixed
        dataframe["energy_cost_load_usd"] = dataframe[energy_data] * dataframe["energy_rate"]
        dic["dataframe"] = dataframe
        dic["list"] = dataframe[["demand_rate_category", "demand_cost_load_usd"]]

    elif tariff_id == "RIT_SC8_Primary_RG&E_2019":
        dataframe["energy_cost_load_usd"] = [0] * len(dataframe.index)
        dataframe["energy_rate_category"] = [0] * len(dataframe.index)
        dataframe["energy_rate"] = [0] * len(dataframe.index)
        dataframe["energy_cost_type"] = "monthly_tod"

        # conditions: assuming flat blended rate
        dataframe["energy_rate"] = 0.09

        # finding energy cost for all data points, then assigning categories
        dataframe["energy_cost_load_usd"] = dataframe[energy_data] * dataframe["energy_rate"]
        dataframe.loc[dataframe["energy_rate"] == 0.09, "energy_rate_category"] = "flat_blended_rate"

        dic["dataframe"] = dataframe
        dic["list"] = dataframe[["demand_rate_category", "demand_cost_load_usd"]]

    elif tariff_id == 'eversource_greater_boston_b3_g6':
        # distribution - transition - revenue decoupling + distribution solar + renewable energy + energy effieciency
        rate = 0.008460 - 0.000520 - 0.000240 + 0.000370 + 0.00500 + 0.010960
        # Monthly projected commodity rates
        commodity_rate = [0.204100, 0.19894, 0.15609, 0.13609, 0.12361, 0.11187, 0.11305, 0.11104, 0.11413,
                          0.12166, 0.13248, 0.16032]
        # assigning energy_rate to dataframe:
        dataframe['energy_rate'] = 0
        dataframe['energy_cost_load_usd'] = 0
        dataframe['energy_rate_category'] = 'flat_rate'
        for month in set(dataframe[date].dt.month):
            dataframe.loc[dataframe[date].dt.month == month, 'energy_rate'] = commodity_rate[month-1] + rate
        # assigning final energy cost for each data point.
        dataframe["energy_cost_load_usd"] = dataframe[energy_data] * dataframe["energy_rate"]
        dic["dataframe"] = dataframe
        dic["list"] = dataframe[["demand_rate_category", "demand_cost_load_usd", date]]

    elif tariff_id == "ConEd_SC8_Demand_Rate1_2019":
        supply_charge = .085
        delivery_charge = .028 + .068
        dataframe['energy_rate'] = supply_charge + delivery_charge
        dataframe['energy_cost_load_usd'] = dataframe['energy_rate'] * dataframe[energy_data]
        dataframe['energy_rate_category'] = 'flat_rate'
        dataframe["energy_cost_type"] = "monthly_tod"
        dic["dataframe"] = dataframe
        dic["list"] = dataframe[["demand_rate_category", "demand_cost_load_usd", date]]

    elif tariff_id == "ConEd_SC8_Rate2_2019":
        dataframe["energy_cost_load_usd"] = [0] * len(dataframe.index)
        dataframe["energy_rate_category"] = [0] * len(dataframe.index)
        dataframe["energy_rate"] = [0] * len(dataframe.index)
        dataframe["energy_cost_type"] = "monthly_tod"

        # conditions: assuming flat blended rate
        dataframe["energy_rate"] = 0.08869 + .02

        # finding energy cost for all data points, then assigning categories
        dataframe["energy_cost_load_usd"] = dataframe[energy_data] * dataframe["energy_rate"]
        dataframe.loc[dataframe["energy_rate"] == 0.08869 + .02, "energy_rate_category"] = "flat_blended_rate"

        dic["dataframe"] = dataframe
        dic["list"] = dataframe[["demand_rate_category1", "demand_cost_load_usd"]]

    elif tariff_id == "ConEd_SC12_Demand_Rate4_2019":
        dataframe['energy_rate'] = 0.085
        dataframe['energy_cost_load_usd'] = dataframe['energy_rate'] * dataframe[energy_data]
        dataframe['energy_rate_category'] = 'flat_rate'
        dataframe["energy_cost_type"] = "daily_tod"
        dic["dataframe"] = dataframe
        dic["list"] = dataframe[["demand_rate_category", "demand_cost_load_usd", date]]

    elif tariff_id == "ConEd_SC8_Demand_Rate4_2019":
        dataframe['energy_rate'] = 0.085
        dataframe['energy_cost_load_usd'] = dataframe['energy_rate'] * dataframe[energy_data]
        dataframe['energy_rate_category'] = 'flat_rate'
        dataframe["energy_cost_type"] = "daily_tod"
        dic["dataframe"] = dataframe
        dic["list"] = dataframe[["demand_rate_category", "demand_cost_load_usd", date]]
    
    elif tariff_id == "flat_rate":
        dataframe['energy_rate'] = 0.0187
        dataframe['energy_cost_load_usd'] = dataframe['energy_rate'] * dataframe[energy_data]
        dataframe['energy_rate_category'] = 'flat_rate'
        dic["dataframe"] = dataframe
        dic["list"] = dataframe[["demand_rate_category", "demand_cost_load_usd", date]]
    return dic

import time
import copy
import pandas as pd
import numpy as np
from demand_tariff import demand_tariff_transform


def tou_peak_shave(capacity_kwh, power_kw, dataframe, max_dod, dis_eff, max_cycles, tariff_id,
                   building_power='original_building_power_kw', power_data='power_post_solar_kw',
                   project_type=None, itc_2019=None):
    dih = (dataframe.date.iloc[1] - dataframe.date.iloc[0])  # data interval hours
    dih = abs(dih.total_seconds() / 3600)
    dataframe['discharge'] = 0
    for month in set(dataframe.date.dt.month):
        dfm = dataframe.loc[dataframe.date.dt.month == month]  # dfm: DataFrame for each Month
        c_prime, value, demand_rate, shave = {}, {}, {}, {}
        for rate_cat in set(dfm.demand_rate_category):
            c_prime[rate_cat] = dfm.loc[dfm.demand_rate_category == rate_cat, 'demand_cost_load_usd'].max()
            value[rate_cat] = dfm.loc[dfm.demand_rate_category == rate_cat, 'demand_rate'].iloc[0]
            demand_rate = copy.deepcopy(value)
        while True:
            for rate_cat in value.keys():
                condition = (dfm.demand_rate_category == rate_cat) & (dfm.demand_cost_load_usd >= c_prime[rate_cat])
                if value[rate_cat] != 0:
                    value[rate_cat] = demand_rate[rate_cat] / dfm.loc[condition].shape[0]
            mx_cat = max(value, key=value.get)
            if (value[mx_cat] == 0) | (dfm.discharge.sum() / dis_eff * dih / capacity_kwh >= max_cycles):
                break
            dfm_r = dfm.loc[dfm.demand_rate_category == mx_cat]
            C = c_prime[mx_cat]
            dfm_r['discharge'] = 0
            dfm_r.loc[dfm_r.demand_cost_load_usd >= C, 'discharge'] = (dfm_r.demand_cost_load_usd - C)/dfm_r.demand_rate
            shave[mx_cat] = dfm_r
            # Calculate resulted discharge energy by day, epd is total energy_per_day:
            dfm.loc[dfm_r.index, 'discharge'] = dfm_r.discharge
            epd = [dfm.loc[dfm.date.dt.day == each_day, 'discharge'].sum() * dih for each_day in set(dfm_r.date.dt.day)]
            # ensuring we're not going above battery limits
            if dfm_r['discharge'].max() > power_kw:
                c_prime[mx_cat] = dfm_r.demand_cost_load_usd.max() - power_kw * dfm_r.demand_rate.iloc[0]
                dfm_r['discharge'] = 0
                dfm_r.loc[dfm_r.demand_cost_load_usd >= c_prime[mx_cat], 'discharge'] = \
                    (dfm_r.demand_cost_load_usd - c_prime[mx_cat]) / dfm_r.demand_rate
                dfm.loc[dfm_r.index, 'discharge'] = dfm_r.discharge
                shave[mx_cat] = dfm_r
                # rendering the value to zero since it exceeded battery limits
                value[mx_cat] = 0
                print('power limit exceeded')
            elif True in (capacity_kwh * dis_eff * max_dod < np.array(epd)):
                c_prime[mx_cat] = c_prime[mx_cat] / .99
                dfm_r['discharge'] = 0
                dfm_r.loc[dfm_r.demand_cost_load_usd >= c_prime[mx_cat], 'discharge'] = \
                    (dfm_r.demand_cost_load_usd - c_prime[mx_cat]) / dfm_r.demand_rate
                dfm.loc[dfm_r.index, 'discharge'] = dfm_r.discharge
                shave[mx_cat] = dfm_r
                # rendering the value to zero since it exceeded battery limits
                value[mx_cat] = 0
                print('energy limit exceeded')
            else:
                c_prime[mx_cat] = c_prime[mx_cat] * .99
        print('now charging for', month)
        # Charging
        dfm['charge'] = 0
        C = dfm.loc[dfm.demand_cost_load_usd > 0, 'demand_cost_load_usd'].min()
        for each_day in set(dfm.date.dt.day):
            dfm.loc[dfm.date.dt.day == each_day, 'charge'] = 0
            daily_discharge = dfm.loc[dfm.date.dt.day == each_day, 'discharge'].sum()

            ppd = daily_discharge / dis_eff
            condition = (dfm.date.dt.day == each_day)
            if itc_2019 == 'yes':
                condition = condition & (dfm.solar_output_kw > 0)

            # condition = condition # & (dfm.demand_cost_load_usd < dfm.demand_cost_load_usd.mean())
            dfm.loc[condition, 'charge'] = ppd / dfm.loc[condition, 'charge'].shape[0]

        dataframe.loc[dataframe.date.dt.month == month, 'charge'] = dfm.charge
        dataframe.loc[dataframe.date.dt.month == month, 'discharge'] = dfm.discharge

        print('for month {} the number of discharges is {}'.format(month, dfm.discharge.sum() * dih / capacity_kwh))

    if project_type == 'solar+storage':
        dataframe['power_post_storage_kw'] = dataframe['power_post_solar_kw'] + dataframe['charge'] - dataframe['discharge']
    else:
        dataframe['power_post_storage_kw'] = dataframe['original_building_power_kw'] + dataframe['charge'] - dataframe['discharge']

    dataframe['energy_post_storage_kwh'] = dataframe['power_post_storage_kw'] * dih
    dataframe['month'] = dataframe.date.dt.month
    dataframe['dps'] = dataframe['power_post_storage_kw'] * dataframe.demand_rate
    dfs = {'shaved_frame': dataframe, 'dis_stats': None}
    return dfs


def storage_peak_shave(capacity_kwh, power_kw, dataframe, max_dod, dis_eff, max_cycles,
                       tariff_id, building_power, itc_2019, power_data='power_post_solar_kw',
                       date='date', project_type='solar+storage'):
    start = time.clock()
    dataframe = copy.deepcopy(dataframe)

    dataframe['month'] = dataframe[date].dt.month
    # Interval of data recordings in hours
    data_interval_hrs = (dataframe[date].iloc[1] - dataframe[date].iloc[0])
    data_interval_hrs = data_interval_hrs.total_seconds() / 3600

    # Storage Peak Shave Function (if this is a solar+storage project then the workable power column
    # to optimize storage should be power post solar otherwise it is original building power)
    if project_type == "solar+storage":
        dataframe["workable_power_kw"] = dataframe[power_data]
        dataframe["demand_cost_post_solar_usd"] = dataframe["workable_power_kw"] * dataframe["demand_rate"]
        dataframe["demand_cost_original_usd"] = dataframe[building_power] * dataframe["demand_rate"]
    elif project_type == 'storage only':
        power_data = building_power
        dataframe["workable_power_kw"] = dataframe[building_power]
        dataframe["demand_cost_original_usd"] = dataframe[building_power] * dataframe["demand_rate"]
    else:
        raise AssertionError('project type is not well defined')
    dataframe["energy_post_storage_kwh"] = dataframe["workable_power_kw"] * data_interval_hrs
    dataframe["power_post_storage_kw"] = dataframe["workable_power_kw"]
    dataframe["demand_cost_post_storage_usd"] = dataframe["power_post_storage_kw"] * dataframe["demand_rate"]

    # Truncating dataframe to match the max number of cycles annually
    dataframe["month_day"] = dataframe[date].dt.month.astype(str) + "_" + dataframe[date].dt.day.astype(str)
    truncated = dataframe.groupby("month_day")["demand_cost_original_usd"].max()
    truncated = truncated.nlargest(n=int(max_cycles), keep="first")
    dataframe_truncated = dataframe.loc[dataframe["month_day"].isin(truncated.index)]

    if (dataframe.loc[0, "peak_demand_cost_type"] == "monthly_peak_tod" or dataframe.loc[
        0, "peak_demand_cost_type"] == "monthly_peak_step"):

        # Discharge Stats------------
        month = []
        monthly_num_discharges = []
        monthly_avg_dispower = []
        monthly_max_dispower = []
        monthly_avg_dod = []
        monthly_max_dod = []
        count = 0

        for m in range(1, 13):
            # Subsets dataframe based on month in for loop
            # Sorts the subset dataframe by Load Cost in descending value
            mframe = dataframe_truncated[dataframe_truncated["month"] == m]
            if len(mframe.index) == 0:
                month.append(m)
                monthly_num_discharges.append(0)
                monthly_avg_dispower.append(0)
                monthly_max_dispower.append(0)
                monthly_avg_dod.append(0)
                monthly_max_dod.append(0)
                continue
            mframe = mframe.sort_values(["demand_cost_post_storage_usd"], ascending=False)

            # Takes the day possessing the peak load cost for the month and subsets mframe by that month_day
            # Sorts the month_day dataframe by load cost in descending value
            # Creates energy shift to help determine how successive the 15min peak is
            peak_day_list = [0]
            peak_day = mframe["month_day"].iloc[0]
            peak_day_list.append(peak_day)

            # Discharge Stats---------------
            average_dis_power_daily = []
            max_dis_power_daily = []
            daily_percent_dod = []

            # While Loop
            # High level: if after discharging the entire capacity for the peak day and resorting the dataframe
            # the peak day is the same stop If after doing the above the peak day becomes a different day
            # we must shave that day to optimize the peak shave for the month and so on
            while (len(peak_day_list) < 3) or (peak_day_list[-1] not in peak_day_list[:-1]):

                dframe = dataframe[dataframe["month_day"] == peak_day_list[-1]]
                dframe = dframe.sort_values(["demand_cost_post_storage_usd"], ascending=False)
                capacity = capacity_kwh * max_dod

                shave_list_kw = []
                area_list_kwh = []
                discharge_times = []
                rate_cat = []

                # Discharge
                # Ok if peak cost trapz are of the same demand_rate otherwise need to modify code
                for i, row in dframe.iterrows():

                    # Discharge
                    if len(shave_list_kw) == 0:
                        shave_list_kw.append(row["power_post_storage_kw"])
                        area_list_kwh.append(0)
                        discharge_times.append(row["date"])
                        rate_cat.append(row["demand_rate_category"])
                    else:
                        shave_list_kw.append(row["power_post_storage_kw"])
                        calc_list = [(x - min(shave_list_kw)) * data_interval_hrs for x in shave_list_kw]
                        area_kwh = sum(calc_list)
                        area_list_kwh.append(area_kwh)
                        discharge_times.append(row["date"])
                        rate_cat.append("demand_rate_category")
                        if area_kwh > capacity:
                            previous_area_kwh = area_list_kwh[-2]
                            area_needed_kwh = capacity - previous_area_kwh
                            shave_line = min(shave_list_kw[:-1]) - (
                                    area_needed_kwh / (len(shave_list_kw[:-1]) * data_interval_hrs))
                            shave_list_kw = shave_list_kw[:-1]
                            discharge_times = discharge_times[:-1]
                            rate_cat = rate_cat[:-1]

                            ##############################################################################

                            test_frame = pd.DataFrame({"discharge_times": discharge_times})
                            test_frame["discharge_times"] = pd.to_datetime(test_frame["discharge_times"])
                            test_frame["power_post_storage_kw"] = shave_list_kw
                            test_frame["demand_rate_category"] = rate_cat
                            test_frame["shave_kw"] = test_frame["power_post_storage_kw"] - shave_line
                            test_frame.loc[test_frame["shave_kw"] > power_kw, "shave_kw"] = power_kw
                            test_frame = test_frame.sort_values(by="discharge_times", ascending=True)
                            test_frame["power_post_storage_kw"] = test_frame["power_post_storage_kw"] - test_frame[
                                "shave_kw"]
                            test_frame["energy_post_storage_kwh"] = test_frame[
                                                                        "power_post_storage_kw"] * data_interval_hrs
                            test_frame["month"] = test_frame["discharge_times"].dt.month
                            test_frame["demand_cost_post_storage_usd"]= demand_tariff_transform(
                                dataframe=test_frame, tariff_id=tariff_id, date='discharge_times',
                                power_data="power_post_storage_kw")[0]["list"]["demand_cost_load_usd"]

                            dataframe.sort_values(by="date", ascending=True)
                            test_frame = test_frame.set_index(
                                dataframe.loc[dataframe["date"].isin(test_frame["discharge_times"])].index)

                            dataframe.loc[
                                dataframe["date"].isin(test_frame["discharge_times"]), ["power_post_storage_kw",
                                                                                        "energy_post_storage_kwh",
                                                                                        "demand_cost_post_storage_usd"]] = \
                                test_frame[["power_post_storage_kw", "energy_post_storage_kwh",
                                            "demand_cost_post_storage_usd"]]
                            break
                            ###################################################################################

                mframe = dataframe[dataframe["month"] == m]
                mframe = mframe.sort_values(["demand_cost_post_storage_usd"], ascending=False)
                peak_day = mframe["month_day"].iloc[0]

                peak_day_list.append(peak_day)

                # Discharge Stats----------------
                list_dis_power = dataframe.loc[dataframe["date"].isin(discharge_times), power_data] - \
                                 dataframe.loc[dataframe["date"].isin(discharge_times), "power_post_storage_kw"]
                list_dis_energy = list_dis_power * data_interval_hrs
                list_dod = list_dis_energy / capacity_kwh

                average_dis_power_daily.append(list_dis_power.mean())
                max_dis_power_daily.append(list_dis_power.max())
                daily_percent_dod.append(list_dod.sum())

                # Charge Logic
                if (len(peak_day_list) > 1) and (peak_day_list[-1] in peak_day_list[:-1]):
                    charge_capacity_kwh = (capacity_kwh * max_dod) / dis_eff
                    charge_line_kw = 0
                    number = 200
                    area_kwh = 0

                    for x in peak_day_list[1:-1]:
                        # print(x)
                        if itc_2019 == "yes":
                            dataframe = dataframe.sort_values(["demand_cost_post_storage_usd"], ascending=True)
                            cframe = dataframe[(dataframe["month_day"] == x) & (dataframe["solar_output_kw"] > 0)]

                            for n in range(number):
                                charge_line_kw = ((cframe["solar_output_kw"].max() / number) * n)
                                charge_list_kw = charge_line_kw - cframe["solar_output_kw"]
                                charge_list_kw[charge_list_kw < 0] = charge_line_kw
                                charge_list_kw[charge_list_kw > power_kw] = power_kw
                                area_kwh = charge_list_kw.sum() * data_interval_hrs
                                if area_kwh >= charge_capacity_kwh:
                                    break

                            if area_kwh < charge_capacity_kwh:
                                print("Not enough solar")

                            cframe["power_post_storage_kw"] = cframe["power_post_storage_kw"] + charge_list_kw
                            cframe["energy_post_storage_kwh"] = cframe["power_post_storage_kw"] * data_interval_hrs
                            dataframe[(dataframe["month_day"] == x) & (dataframe["solar_output_kw"] > 0)] = cframe

                        else:
                            dataframe = dataframe.sort_values(["demand_cost_post_storage_usd"], ascending=True)
                            cframe = dataframe[dataframe["month_day"] == x]

                            for n in range(number):
                                charge_line_kw = (((cframe["power_post_storage_kw"].max() - cframe[
                                    "power_post_storage_kw"].min()) / number) * n) + cframe[
                                                     "power_post_storage_kw"].min()
                                charge_list_kw = charge_line_kw - cframe.loc[
                                    cframe["power_post_storage_kw"] <= charge_line_kw, "power_post_storage_kw"]
                                area_kwh = charge_list_kw.sum() * data_interval_hrs
                                if area_kwh >= charge_capacity_kwh:
                                    break
                            criteria1 = (cframe["power_post_storage_kw"] <= charge_line_kw) & (
                                    (charge_line_kw - cframe["power_post_storage_kw"]) <= power_kw)
                            criteria2 = (cframe["power_post_storage_kw"] <= charge_line_kw) & (
                                    (charge_line_kw - cframe["power_post_storage_kw"]) > power_kw)
                            cframe.loc[criteria1, "power_post_storage_kw"] = charge_line_kw
                            cframe.loc[criteria2, "power_post_storage_kw"] = cframe.loc[
                                                                                 criteria2, "power_post_storage_kw"] + power_kw
                            cframe["energy_post_storage_kwh"] = cframe["power_post_storage_kw"] * data_interval_hrs
                            dataframe[dataframe["month_day"] == x] = cframe

                    dataframe = dataframe.sort_values(by="date")
                    count = count + 1
                    month.append(m)
                    monthly_num_discharges.append(len(peak_day_list[:-2]))
                    monthly_avg_dispower.append(np.mean(average_dis_power_daily))
                    monthly_max_dispower.append(max(max_dis_power_daily))
                    monthly_avg_dod.append(np.mean(daily_percent_dod))
                    monthly_max_dod.append(max(daily_percent_dod))
                    # print("Percent Complete: ", round(count / 12, 2))

        # Discharge Stats-------------
        dis_stats = pd.DataFrame({"month": month, "monthly_num_discharges": monthly_num_discharges,
                                  "monthly_avg_discharge_power_kw": monthly_avg_dispower,
                                  "monthly_max_discharge_power_kw": monthly_max_dispower,
                                  "monthly_avg_dod_%": monthly_avg_dod, "monthly_max_dod_%": monthly_max_dod})
        dfs = {'shaved_frame': dataframe, 'dis_stats': dis_stats}
        # print("Program Total Time (seconds):", time.clock() - start)
        return dfs

    """
    # TODO: change the code
    elif (dataframe.loc[0, "peak_demand_cost_type"] == "daily_peak_tod" or dataframe.loc[
        0, "peak_demand_cost_type"] == "daily_peak_step"):
        # Discharge Stats------------
        month = []
        monthly_num_discharges = []
        monthly_avg_dispower = []
        monthly_max_dispower = []
        monthly_avg_dod = []
        monthly_max_dod = []
        count = 0

        for m in range(1, 13):
            # Subsets dataframe based on month in for loop
            # Sorts the subset dataframe by Load Cost in descending value
            mframe = dataframe_truncated[dataframe_truncated["month"] == m]
            if len(mframe.index) == 0:
                month.append(m)
                monthly_num_discharges.append(0)
                monthly_avg_dispower.append(0)
                monthly_max_dispower.append(0)
                monthly_avg_dod.append(0)
                monthly_max_dod.append(0)
                continue
            mframe = mframe.sort_values(["demand_cost_post_storage_usd"], ascending=False)

            monthly_cycle_count = 0
            # Discharge Stats---------------
            average_dis_power_daily = []
            max_dis_power_daily = []
            daily_percent_dod = []

            # While Loop
            # High level: if after discharging the entire capacity for the peak day and resorting the dataframe
            # the peak day is the same stop. If after doing the above the peak day becomes a different day
            # we must shave that day to optimize the peak shave for the month and so on
            for j in mframe["month_day"]:
                dframe = dataframe[dataframe["month_day"] == j]
                dframe = dframe.sort_values(["demand_cost_post_storage_usd"], ascending=False)
                capacity = capacity_kwh

                shave_list_kw = []
                area_list_kwh = []
                discharge_times = []

                # Discharge
                # Ok if peak cost trapz are of the same demand_rate otherwise need to modify code
                for i, row in dframe.iterrows():
                    for_loop_start_time = time.clock()
                    if len(shave_list_kw) == 0:
                        shave_list_kw.append(row["power_post_storage_kw"])
                        area_list_kwh.append(0)
                        discharge_times.append(row["date"])
                    else:
                        shave_list_kw.append(row["power_post_storage_kw"])
                        calc_list = [(x - min(shave_list_kw)) * data_interval_hrs for x in shave_list_kw]
                        area_kwh = sum(calc_list)
                        area_list_kwh.append(area_kwh)
                        discharge_times.append(row["date"])
                        if area_kwh > capacity_kwh:
                            previous_area_kwh = area_list_kwh[-2]
                            area_needed_kwh = capacity_kwh - previous_area_kwh
                            shave_line = min(shave_list_kw[:-1]) - (
                                    area_needed_kwh / (len(shave_list_kw[:-1]) * data_interval_hrs))
                            shave_list_kw = shave_list_kw[:-1]
                            discharge_times = discharge_times[:-1]

                            ##############################################################################

                            test_frame = pd.DataFrame({"discharge_times": discharge_times})
                            test_frame["discharge_times"] = pd.to_datetime(test_frame["discharge_times"])
                            test_frame["power_post_storage_kw"] = shave_list_kw
                            test_frame["shave_kw"] = test_frame["power_post_storage_kw"] - shave_line
                            test_frame.loc[test_frame["shave_kw"] > power_kw, "shave_kw"] = power_kw
                            test_frame = test_frame.sort_values(by="discharge_times", ascending=True)
                            test_frame["power_post_storage_kw"] = test_frame["power_post_storage_kw"] - test_frame[
                                "shave_kw"]
                            test_frame["energy_post_storage_kwh"] = test_frame[
                                                                        "power_post_storage_kw"] * data_interval_hrs
                            test_frame["month"] = test_frame["discharge_times"].dt.month

                            # calling demand transform function
                            test_frame["demand_cost_post_storage_usd"] = demand_tariff_transform(
                                dataframe=test_frame, tariff_id=tariff_id, date='discharge_times',
                                power_data='power_post_storage_kw')["list"]["demand_cost_load_usd"]

                            dataframe.sort_values(by="date", ascending=True)
                            test_frame = test_frame.set_index(
                                dataframe.loc[dataframe["date"].isin(test_frame["discharge_times"])].index)

                            dataframe.loc[
                                dataframe["date"].isin(test_frame["discharge_times"]), ["power_post_storage_kw",
                                                                                       "energy_post_storage_kwh",
                                                                                       "demand_cost_post_storage_usd"]] = \
                            test_frame[
                                ["power_post_storage_kw", "energy_post_storage_kwh", "demand_cost_post_storage_usd"]]
                            break
                            ###################################################################################

                    # Discharge Stats----------------
                    list_dis_power = dataframe.loc[dataframe["date"].isin(discharge_times), building_power] - \
                                    dataframe.loc[dataframe["date"].isin(discharge_times), "power_post_storage_kw"]
                    list_dis_energy = list_dis_power * data_interval_hrs
                    list_dod = list_dis_energy / capacity_kwh

                    average_dis_power_daily.append(list_dis_power.mean())
                    max_dis_power_daily.append(list_dis_power.max())
                    daily_percent_dod.append(list_dod.sum())

                if (len(peak_day_list) > 3) and (peak_day_list[-1] in peak_day_list[:-1]):
                    count = count + 1
                    monthly_cycle_count = monthly_cycle_count + 1
                    month.append(m)
                    monthly_num_discharges.append(monthly_cycle_count)
                    monthly_avg_dispower.append(np.mean(average_dis_power_daily))
                    monthly_max_dispower.append(max(max_dis_power_daily))
                    monthly_avg_dod.append(np.mean(daily_percent_dod))
                    monthly_max_dod.append(max(daily_percent_dod))
                    print("Percent Complete: ", round(count / max_cycles, 2))

        # Discharge Stats-------------
        dis_stats = pd.DataFrame({"month": month, "monthly_num_discharges": monthly_num_discharges,
                                  "monthly_avg_discharge_power_kw": monthly_avg_dispower,
                                  "monthly_max_discharge_power_kw": monthly_max_dispower,
                                  "monthly_avg_dod_%": monthly_avg_dod, "monthly_max_dod_%": monthly_max_dod})
        dfs = {"shaved_frame": dataframe, "dis_stats": dis_stats}
        print("Program Total Time (seconds):", time.clock() - start)
        return dfs
    """
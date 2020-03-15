import pandas as pd
from demand_tariff import demand_tariff_transform
from energy_tariff import energy_tariff_transform


def demand_savings(data, tariff_id, demand_tariff_type="monthly_peak_tod", project_type='solar+storage'):
    # data = dic["shaved_frame"]

    
    """
        if project_type == "solar+storage":
            data.loc[data["power_post_solar_kw"] < 0, "power_post_solar_kw"] = 0
        data.loc[data["power_post_storage_kw"] < 0, "power_post_storage_kw"] = 0
        data.loc[data["power_post_storage_kw"] < 0, "power_post_storage_kw"] = 0
    """
    if demand_tariff_type == "monthly_peak_tod":
        monthly_power_costs = pd.DataFrame({"month": list(range(1, 13))})

        test_demand_rate_categories = data["demand_rate_category"]
        test_demand_rate_categories = test_demand_rate_categories.unique()
        demand_rate_categories = [x for x in test_demand_rate_categories if isinstance(x, str)]

        if project_type == "solar+storage":
            solar_peak_dataset = "power_post_solar_kw"
        original_peak_dataset = "original_building_power_kw"
        final_peak_dataset = "power_post_storage_kw"

        for i in range(len(demand_rate_categories)):
            monthly_power_costs[original_peak_dataset + "_" + demand_rate_categories[i]] = [0] * len(
                monthly_power_costs.index)
            if project_type == "solar+storage":
                monthly_power_costs[solar_peak_dataset + "_" + demand_rate_categories[i]] = [0] * len(
                    monthly_power_costs.index)
            monthly_power_costs[final_peak_dataset + "_" + demand_rate_categories[i]] = [0] * len(
                monthly_power_costs.index)
            monthly_power_costs[demand_rate_categories[i] + "_" + "demand_rate"] = [0] * len(monthly_power_costs.index)

        for m in range(1, 13):
            # Month Shaved Frame
            m_shaved_frame = data[data["month"] == m]
            for i in demand_rate_categories:
                # Peak Cost Type Shaved Frame

                if len(m_shaved_frame[m_shaved_frame["demand_rate_category"] == i].index) == 0:
                    continue
                else:
                    monthly_power_costs.loc[monthly_power_costs["month"] == m, original_peak_dataset + "_" + i] = \
                        m_shaved_frame.loc[m_shaved_frame["demand_rate_category"] == i, original_peak_dataset].max()

                    if project_type == "solar+storage":
                        monthly_power_costs.loc[monthly_power_costs["month"] == m, solar_peak_dataset + "_" + i] = \
                            m_shaved_frame.loc[m_shaved_frame["demand_rate_category"] == i, solar_peak_dataset].max()

                    monthly_power_costs.loc[monthly_power_costs["month"] == m, final_peak_dataset + "_" + i] = \
                        m_shaved_frame.loc[m_shaved_frame["demand_rate_category"] == i, final_peak_dataset].max()

                    monthly_power_costs.loc[monthly_power_costs["month"] == m, i + "_" + "demand_rate"] = \
                        m_shaved_frame.loc[m_shaved_frame["demand_rate_category"] == i, "demand_rate"].iloc[0]

        monthly_power_costs.fillna(0, inplace=True)

        for i in demand_rate_categories:
            if demand_tariff_type == "monthly_peak_tod":
                if project_type == "solar+storage":
                    monthly_power_costs[i + "_" + "monthly_post_solar_savings_usd"] = \
                        (monthly_power_costs[original_peak_dataset + "_" + i] -
                         monthly_power_costs[solar_peak_dataset + "_" + i]) * monthly_power_costs[
                            i + "_" + "demand_rate"]

                monthly_power_costs[i + "_" + "monthly_post_storage_savings_usd"] \
                    = (monthly_power_costs[original_peak_dataset + "_" + i] -
                       monthly_power_costs[final_peak_dataset + "_" + i]) * monthly_power_costs[i + "_" + "demand_rate"]
            elif demand_tariff_type == "monthly_peak_step":
                # print(monthly_power_costs)
                original = demand_tariff_transform(
                    monthly_power_costs, tariff_id,
                    power_data=original_peak_dataset + "_" + i)["list"]["demand_cost_load_usd"]

                if project_type == "solar+storage":
                    solar = demand_tariff_transform(
                        monthly_power_costs, tariff_id,
                        power_data=solar_peak_dataset + "_" + i)["list"]["demand_cost_load_usd"]
                    monthly_power_costs[i + "_" + "monthly_post_solar_savings_usd"] = (original - solar)

                final = demand_tariff_transform(
                    monthly_power_costs, tariff_id,
                    power_data=final_peak_dataset + "_" + i)["list"]["demand_cost_load_usd"]
                monthly_power_costs[i + "_" + "monthly_post_storage_savings_usd"] = (original - final)

        for _ in demand_rate_categories:
            monthly_power_costs["monthly_total_post_solar_demand_savings_usd"] = [0] * len(monthly_power_costs.index)
        monthly_power_costs["monthly_total_post_storage_demand_savings_usd"] = [0] * len(monthly_power_costs.index)

        for i in demand_rate_categories:
            if project_type == "solar+storage":
                monthly_power_costs["monthly_total_post_solar_demand_savings_usd"] = \
                    monthly_power_costs["monthly_total_post_solar_demand_savings_usd"] + \
                    monthly_power_costs[i + "_" + "monthly_post_solar_savings_usd"]

            monthly_power_costs["monthly_total_post_storage_demand_savings_usd"] = \
                monthly_power_costs["monthly_total_post_storage_demand_savings_usd"] + \
                monthly_power_costs[i + "_" + "monthly_post_storage_savings_usd"]

        return monthly_power_costs

    elif (demand_tariff_type == "daily_peak_tod" or
          demand_tariff_type == "daily_peak_step"):
        # TODO: need to code daily demand
        return []


def energy_savings(data, tariff_id, demand_tariff_type='monthly_peak_tod',project_type='solar+storage'):

    monthly_energy_costs = pd.DataFrame({"month": list(range(1, 13))})
    test_energy_rate_categories = data["energy_rate_category"]
    test_energy_rate_categories = test_energy_rate_categories.unique()
    energy_rate_categories = [x for x in test_energy_rate_categories if isinstance(x, str)]
    original_load_dataset = "original_building_energy_kwh"
    final_load_dataset = "energy_post_storage_kwh"

    if project_type == "solar+storage":
        solar_load_dataset = "energy_post_solar_kwh"

    for i in range(len(energy_rate_categories)):
        monthly_energy_costs[original_load_dataset + "_" + energy_rate_categories[i]] = [0] * len(
            monthly_energy_costs.index)
        monthly_energy_costs[final_load_dataset + "_" + energy_rate_categories[i]] = [0] * len(
            monthly_energy_costs.index)
        monthly_energy_costs[energy_rate_categories[i] + "_" + "energy_rate"] = [0] * len(monthly_energy_costs.index)

    # Calculates energy usage per month per rate category
    for m in range(1, 13):
        # Month Shaved Frame
        m_shaved_frame = data[data["month"] == m]
        for i in energy_rate_categories:
            # Load Cost Type Shaved Frame

            if len(m_shaved_frame[m_shaved_frame["energy_rate_category"] == i].index) == 0:
                continue
            else:
                monthly_energy_costs.loc[monthly_energy_costs["month"] == m, original_load_dataset + "_" + i] = \
                    m_shaved_frame.loc[m_shaved_frame["energy_rate_category"] == i, original_load_dataset].sum()

                if project_type == "solar+storage":
                    monthly_energy_costs.loc[monthly_energy_costs["month"] == m, solar_load_dataset + "_" + i] = \
                        m_shaved_frame.loc[m_shaved_frame["energy_rate_category"] == i, solar_load_dataset].sum()

                monthly_energy_costs.loc[monthly_energy_costs["month"] == m, final_load_dataset + "_" + i] = \
                    m_shaved_frame.loc[m_shaved_frame["energy_rate_category"] == i, final_load_dataset].sum()

                monthly_energy_costs.loc[monthly_energy_costs["month"] == m, i + "_" + "energy_rate"] = \
                    m_shaved_frame.loc[m_shaved_frame["energy_rate_category"] == i, "energy_rate"].iloc[0]

    monthly_energy_costs.fillna(0, inplace=True)

    # Calculates energy savings per month post storage and post solar per rate category
    for i in energy_rate_categories:
        if demand_tariff_type == "monthly_peak_tod":

            if project_type == "solar+storage":
                monthly_energy_costs[i + "_" + "post_solar_monthly_savings_usd"] = \
                    (monthly_energy_costs[original_load_dataset + "_" + i] -
                     monthly_energy_costs[solar_load_dataset + "_" + i]) * \
                    monthly_energy_costs[i + "_" + "energy_rate"]

            monthly_energy_costs[i + "_" + "post_storage_monthly_savings_usd"] = \
                (monthly_energy_costs[original_load_dataset + "_" + i] -
                 monthly_energy_costs[final_load_dataset + "_" + i]) * monthly_energy_costs[i + "_" + "energy_rate"]

        elif data.loc[0, "energy_cost_type"] == "monthly_step":
            # print(monthly_energy_costs)
            original = energy_tariff_transform(
                monthly_energy_costs, tariff_id,
                energy_data=original_load_dataset + "_" + i)["list"]["energy_cost_load_usd"]

            solar = energy_tariff_transform(
                monthly_energy_costs, tariff_id,
                energy_data=solar_load_dataset + "_" + i)["list"]["energy_cost_load_usd"]

            final = energy_tariff_transform(
                monthly_energy_costs, tariff_id,
                energy_data=final_load_dataset + "_" + i)["list"]["energy_cost_load_usd"]

            if project_type == "solar+storage":
                monthly_energy_costs[i + "_" + "post_solar_monthly_savings_usd"] = original - solar
            monthly_energy_costs[i + "_" + "post_storage_monthly_savings_usd"] = original - final

    if project_type == "solar+storage":
        monthly_energy_costs["post_solar_monthly_total_energy_savings_usd"] = [0] * len(monthly_energy_costs.index)
    monthly_energy_costs["post_storage_monthly_total_energy_savings_usd"] = [0] * len(monthly_energy_costs.index)

    for i in energy_rate_categories:
        if project_type == "solar+storage":
            monthly_energy_costs["post_solar_monthly_total_energy_savings_usd"] = \
                monthly_energy_costs["post_solar_monthly_total_energy_savings_usd"] + \
                monthly_energy_costs[i + "_" + "post_solar_monthly_savings_usd"]

        monthly_energy_costs["post_storage_monthly_total_energy_savings_usd"] = \
            monthly_energy_costs["post_storage_monthly_total_energy_savings_usd"] + \
            monthly_energy_costs[i + "_" + "post_storage_monthly_savings_usd"]

    return monthly_energy_costs

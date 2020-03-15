import pandas as pd
import copy
import time
import helper
import os
from os.path import join
from demand_tariff import demand_tariff_transform
from energy_tariff import energy_tariff_transform
from savings import demand_savings, energy_savings
from optimizer import storage_peak_shave, tou_peak_shave
import vder_optimization
from vder_optimization import optimize_df


class Calculator:

    def __init__(self, df, tariff_id, date='date'):
        self.df = copy.deepcopy(df)
        self.tariff_id = tariff_id
        self.date = date
        self.dic = {}
        self.solar_data_transform = None
        self.storage_only_transform = None
        self.project_type = None
        self.solar_data = None
        self.dtt = None
        self.inputs = {'Date': time.strftime('%b-%d-%Y'), 'Calculator Version': 1.3, 'tariff_id': tariff_id}
        self.dih = helper.df_interval_hours(self.df)

    def add_solar(self, solar, hourly_intervals, mode='safe'):
        self.storage_only_transform = None
        if mode == 'safe':
            solar_data = helper.add_pv_to_df(self.df, solar, power_data='original_building_power_kw')
        elif mode == 'fast':
            solar_data = helper.add_solar_to_df(self.df, solar, hourly_intervals)
            

        # TODO: add power post solar here 
        # apply df to demand_tariff_transform, dtt: demand tariff type
        solar_data_transform, self.dtt = demand_tariff_transform(dataframe=solar_data, tariff_id=self.tariff_id,
                                                                 date=self.date, power_data='power_post_solar_kw')
        # apply df to energy_tariff_transform
        self.solar_data_transform = energy_tariff_transform(solar_data_transform['dataframe'],
                                                            tariff_id=self.tariff_id, date=self.date,
                                                            energy_data='energy_post_solar_kwh')
        
        return

    @property
    def solar_df(self):
        return self.solar_data

    def add_storage_only(self):
        self.solar_data_transform = None
        d_transform, self.dtt = demand_tariff_transform(dataframe=self.df, tariff_id=self.tariff_id,
                                                        date=self.date, power_data='original_building_power_kw')

        self.storage_only_transform = energy_tariff_transform(dataframe=d_transform['dataframe'],
                                                              tariff_id=self.tariff_id, date=self.date,
                                                              energy_data='original_building_energy_kwh')

    def peak_shave(self, power_kw, capacity_kwh, dis_eff=.85, max_dod=.85, max_cycles=300, itc_2019='no'):
        self.inputs.update({'Capacity kWh': capacity_kwh, 'Power kW': power_kw, 'Max DOD': max_dod,
                            'Max Cycles': max_cycles, 'Discharge Efficiency': dis_eff, 'ITC Incentive': itc_2019})
        print(self.dtt)
        if self.dtt == 'monthly_peak_tod':
            f = tou_peak_shave
        else:
            f = storage_peak_shave
        # solar + storage
        if self.solar_data_transform is not None:
            self.project_type = 'solar+storage'
            self.dic = f(capacity_kwh=capacity_kwh, power_kw=power_kw,
                         dataframe=self.solar_data_transform['dataframe'],
                         dis_eff=dis_eff, max_dod=max_dod, max_cycles=max_cycles,
                         tariff_id=self.tariff_id,
                         building_power='original_building_power_kw', power_data='power_post_solar_kw',
                         project_type='solar+storage', itc_2019=itc_2019)

            d_savings = demand_savings(self.dic['shaved_frame'], tariff_id=self.tariff_id, demand_tariff_type=self.dtt,
                                       project_type='solar+storage')
            e_savings = energy_savings(self.dic['shaved_frame'], tariff_id=self.tariff_id, demand_tariff_type=self.dtt,
                                       project_type='solar+storage')

        # storage  only
        elif self.storage_only_transform is not None:
            self.project_type = 'storage only'
            self.dic = f(capacity_kwh=capacity_kwh, power_kw=power_kw,
                         dataframe=self.storage_only_transform['dataframe'],
                         dis_eff=dis_eff, max_dod=max_dod, max_cycles=max_cycles,
                         tariff_id=self.tariff_id,
                         building_power='original_building_power_kw',
                         project_type='storage only', itc_2019=itc_2019)
            d_savings = demand_savings(self.dic['shaved_frame'], tariff_id=self.tariff_id, project_type='storage only')
            e_savings = energy_savings(self.dic['shaved_frame'], tariff_id=self.tariff_id, project_type='storage only')

        return d_savings, e_savings

    #def optimize_year(df, power, capacity, max_discharges, eff=.8, project_type='storage only', itc=False):
    def vder_peak_shave(self, power_kw, capacity_kwh, max_discharges, eff=.8, itc=False):
        if self.solar_data_transform is not None:
            self.project_type = 'solar+storage'
            dataframe = self.solar_data_transform['dataframe']
        elif self.storage_only_transform is not None:
            self.project_type = 'storage only'
            dataframe = self.storage_only_transform['dataframe']

        shaved_frame, _ = optimize_df(df=dataframe, power=power_kw,
                                      capacity=capacity_kwh, max_discharges=max_discharges, eff=eff, itc=itc)


        """
        power_post_storage_kw, energy_post_storage_kwh, month, 
        dataframe['dps'] = dataframe['power_post_storage_kw'] * dataframe.demand_rate
        """

    def to_excel(self, file_name):
        assert file_name, 'file_name is not provided'
        self.inputs['Project Type'] = self.project_type
        solar_dic = helper.load_data('solar_inputs')
        if self.project_type == 'solar+storage':
            self.inputs.update(solar_dic)
        elif self.project_type == 'storage only':
            for key in solar_dic.keys():
                self.inputs.pop(key, None)
        else:
            raise NameError('Unknown project type')
        directory = os.path.dirname(os.path.abspath(__file__))
        directory = join(directory, "proposals", file_name + '.xlsx')
        writer = pd.ExcelWriter(directory, engine='xlsxwriter')

        self.dic["shaved_frame"].to_excel(writer, sheet_name='Peak Shave Dataframe')
        self.dic["dis_stats"].to_excel(writer, sheet_name='Discharge Statistics')
        demand_savings(self.dic["shaved_frame"], self.tariff_id, self.project_type).to_excel(writer,
                                                                                             sheet_name='Demand Savings')
        energy_savings(self.dic["shaved_frame"], self.tariff_id, self.project_type).to_excel(writer,
                                                                                             sheet_name='Energy Savings')
        pd.DataFrame(data=[list(self.inputs.values())], columns=list(self.inputs.keys())).T.to_excel(writer,
                                                                                                     sheet_name='Inputs')
        writer.save()
        return

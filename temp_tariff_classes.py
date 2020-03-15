class Tariff:

    def __init__(self, demand, month=None, energy_rate=0, fixed_power_rate=0, time_resolution=15):
        """
        Parameters
        ---------
        demand : pandas.DataFrame, power demand in kw
        month : pandas.DataFrame, either the real month or billing month
        energy_rate : float, energy rate in $/kwh
        fixed_power_rate : float, peak power rate in $/kw
        time_resolution: int, time resolution of demand DataFrame in minutes
        """
        self.demand = demand
        self.month = month
        self.energy_rate = energy_rate
        self.fixed_power_rate = fixed_power_rate
        self.ts = time_resolution
        self.energy = demand * time_resolution / 60

        # assert isinstance(demand, pd.DataFrame)
        # assert  isinstance(month, pd.DataFrame)

    @property
    def fixed_energy_cost_monthly(self):
        cost = {}
        for each_month in set(self.month):
            cost[each_month] = self.energy[self.month == each_month].sum() * self.energy_rate
        return cost

    @property
    def fixed_power_cost_monthly(self):
        cost = {}
        for each_month in set(self.month):
            cost[each_month] = self.demand[self.month == each_month].max() * self.fixed_power_rate
        return cost

    @property
    def fixed_rate_monthly_bill(self):
        cost = {}
        for each_month in set(self.month):
            cost[each_month] = self.energy[self.month == each_month].sum() * self.energy_rate \
                               + self.demand[self.month == each_month].max() * self.fixed_power_rate
        return cost


class Tiered(Tariff):
    """
    Calculating electricity cost for tiered tariffs.
    if the tariff is structured as follows:
        0-100kwh: $0.1/kwh
        100-200kwh: $0.2/kwh
        200-300kwh: $0.3/kwh
        300 and above: $0.5/kwh
    then:
        tiers = [0, 100, 200, 300]
        rates = [.1, .2, .3, .5]
    """

    def __init__(self, tiers, rates, demand,
                 month=None, billing='monthly', time_resolution=15, fixed_power_rate=0):
        Tariff.__init__(self, demand, month, time_resolution, fixed_power_rate)
        self.tiers = tiers
        self.rates = rates
        self.billing = billing

        assert len(tiers) == len(rates), " tier and rate levels don't match"

    @property
    def tiered_energy_cost_monthly(self):
        tier_payments = [0]
        for i in range(1, len(self.tiers)):
            tier_payments.append(tier_payments[-1] + (self.tiers[i] - self.tiers[i - 1]) * self.rates[i - 1])
        # TODO add option for daily bills.
        if self.billing == 'monthly':
            assert self.month.any(), 'month is not provided'
            cost = {}
            for each_month in set(self.month):
                e = self.energy[self.month == each_month].sum()
                for i in reversed(range(len(self.tiers))):
                    if e >= self.tiers[i]:
                        cost[each_month] = (e - self.tiers[i]) * self.rates[i] + tier_payments[i]
                        break
        return cost


class TOU1:

    def __init__(self, df):
        assert isinstance(df, pd.DataFrame)
        assert 'date' in df.columns, 'data frame has no date column, either add date column or rename it to date'
        assert 'power' in df.columns, 'data frame has no power column, add or rename'
        # assert 'hour' in df.columns, 'data frame has no hour column, add or rename'
        assert 'energy' in df.columns, 'data frame has no energy column, add or rename'

        self.df = df
        self.power = df.power
        self.energy = df.energy

    def m(self, *args):
        """

        t1 = {'month': [1,2,3,5], 'e_tiers': [0,100,200,500], 'e_rates': [.1,.2,.3,.5],
        'p_tiers': [0,1000], 'p_rates':[0,.1], 'p_flat': [100, 0]}
        t2 = {'month': [4,6,7,8,9,10,11,12], 'tiers': [0,200], 'rates': [.1,.15]}
        :param args:
        :return:
        """
        cost = {}
        p_cost = {}
        for dic in args:
            df = self.df[self.df.date.dt.month.isin(dic['month'])]
            cost.update(energy_cost(df, dic['e_tiers'], dic['e_rates']))
            p_cost.update(power_cost(df, dic['p_rate']))
        return cost, p_cost

    def w():
        pass

    def wm():
        pass

    def wmh():
        pass


# TODO: change the code to make it tell between kw and kwh
# TODO: recode energy_cost and power_cost
"""
def energy_cost(df, tiers, rates):
    month = df.date.dt.month
    tier_payments = [0]
    for i in range(1, len(tiers)):
        tier_payments.append(tier_payments[-1] + (tiers[i] - tiers[i - 1]) * rates[i - 1])
    # TODO add option for daily bills.
    cost = {}
    for each_month in set(month):
        e = df.energy[month == each_month].sum()
        for i in reversed(range(len(tiers))):
            if e >= tiers[i]:
                cost[each_month] = (e - tiers[i]) * rates[i] + tier_payments[i]
                break
    return cost


def power_cost(df, rate):
    month = df.date.dt.month
    cost = {}
    for each_month in set(month):
        cost[each_month] = df.power[month == each_month].max() * rate
    return cost
"""


class TOU:

    def __init__(self, df):
        assert isinstance(df, pd.DataFrame)
        assert 'date' in df.columns, 'data frame has no date column, add or rename'
        assert 'hour' in df.columns, 'data frame has no hour column, add or rename'
        assert 'power_kw' in df.columns, 'data frame has no power column, add or rename'
        assert 'energy_kw' in df.columns, 'data frame has no energy column, add or rename'

        self.df = df
        self.power_kw = df.power_kw
        self.energy_kwh = df.energy_kwh

    def power_rates(self, *args):
        """
        example:
        peak = {'name': 'peak', 'month': [6,7,8], 'weekday': ['Friday', 'Saturday', 'Sunday'], 'hour': list(range(18,23)),
                'p_rate': 30}

        mid_peak = {'name': 'mid_peak', 'month': [6,7,8], 'weekday': ['Friday', 'Saturday', 'Sunday'],
                    'hour': list(range(13,18)), 'p_rate': 20}

        off_peak = {'name': 'off_peak', 'p_rate': 10}

        >>> power_rates(df, peak, mid_peak, off_peak)
        :return:
        :param args: set off dictionaries containing the structure of the tariff
        :return: 1) dictionary for each time period.
                 2) DataFrame with additional column p_rate that has the maximum power charge for the time period.
        """
        peaks = []
        tou_dic = {}
        for dic in args:
            assert 'name' in dic.keys(), "'name' is not assigned to one of the dictionaries"
            df = copy.deepcopy(self.df)
            if "month" in dic.keys():
                df = df[df.date.dt.month.isin(dic['month'])]
            if "weekday" in dic.keys():
                df = df[df.date.dt.day_name().isin(dic['weekday'])]
            if "hour" in dic.keys():
                df = df[df.hour.isin(dic['hour'])]
            if ("month" in dic.keys()) | ("weekday" in dic.keys()) | ("hour" in dic.keys()):
                df['p_rate'] = dic['p_rate']
                tou_dic[dic['name']] = df
                peaks.append(df)

        for dic in args:
            if ("month" in dic.keys()) | ("weekday" in dic.keys()) | ("hour" in dic.keys()):
                pass
            else:
                df = copy.deepcopy(self.df)
                df_peaks = pd.concat(peaks)
                df_rest = df[~df.index.isin(df_peaks.index)]
                df_rest = copy.deepcopy(df_rest)
                df_rest['p_rate'] = dic['p_rate']
                tou_dic[dic['name']] = df_rest
                peaks.append(df_rest)
        return tou_dic, pd.concat(peaks, sort=True).sort_index()


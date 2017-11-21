"""
EV advise unit
"""
import argparse
import datetime
import random
from itertools import zip_longest

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as py
import pytz
import yaml
from tqdm import tqdm

import pricing
from battery import Charger as ChargerClass
from house import IEC
from ier.ier import IER
from safehdf import SafeHDF5Store

# a global charger to take advantage of result caching
Charger = ChargerClass()


def round_time(dt: datetime.datetime, dateDelta: datetime.timedelta):
    """Round a datetime object to any time laps in seconds
    dt : datetime.datetime object.
    roundTo : Closest number of seconds to round to
    """
    roundTo = dateDelta.total_seconds()
    seconds = (dt - dt.min).seconds
    rounding = (seconds + roundTo / 2) // roundTo * roundTo
    return dt + datetime.timedelta(0, rounding - seconds, -dt.microsecond)


def random_time(mean: datetime.time, std: datetime.timedelta, date: datetime.date = None):
    """Generates a random time object using a gaussian distribution
    :param mean: in minutes
    :param std: in minutes
    :param date: (optional) if given, a datetime object will be returned
    :return: a random time object (datetime if date is provided)
    """
    if date is None:
        return (datetime.datetime.combine(datetime.date(1, 1, 1), mean) + datetime.timedelta(
            seconds=np.random.normal(0, std.total_seconds()))).time()
    else:
        return datetime.datetime.combine(date, mean) + datetime.timedelta(
            seconds=np.random.normal(0, std.total_seconds()))


def calc_charge(action, interval, cur_charge):
    # Given that Charging rates are in kW and self.interval is in minutes, returns joules

    Charger.set_charge(cur_charge)

    return Charger.charge(action, interval)


def calc_charge_with_error(action, interval, cur_charge):
    # Given that Charging rates are in kW and self.interval is in minutes, returns joules

    Charger.set_charge(cur_charge)

    current_charge, battery_consumption = Charger.charge(action, interval)

    if current_charge != cur_charge:
        current_charge += np.random.normal(0, 0.05 * abs((current_charge - cur_charge)) / 3)

    return current_charge, battery_consumption


class Node:
    def __init__(self, battery, time):
        self.battery = battery
        self.time = time

    def __hash__(self):
        return hash((self.battery, self.time))

    def __eq__(self, other):
        return isinstance(other, self.__class__) \
               and self.battery == other.battery \
               and self.time == other.time

    def __repr__(self):
        return "{0}-{1}".format(self.time, self.battery)


class EVPlanner:
    _name = 'Not Implemented'

    def __init__(self, data, current_time: datetime.datetime, end_time: datetime.datetime, current_battery,
                 target_battery,
                 interval: datetime.timedelta, action_set, starting_max_demand, pricing_model: pricing.PricingModel):
        self._data = data

        self.current_time = current_time
        self.end_time = end_time

        self.current_battery = current_battery
        self.target_battery = target_battery

        self.interval = interval

        self.action_set = action_set

        self.starting_max_demand = starting_max_demand

        # this is the return index (a pandas datetime index from now to end time with interval for freq)
        self.result_index = pd.date_range(current_time, end_time - interval, freq=interval)

        # pricing model
        self._pricing_model = pricing_model

    def advise(self):
        """
        :return: returns a list of actions. should have one action per interval from start time to end time
        """
        pass


class EVA(EVPlanner):
    _name = 'SmartCharge'

    def __init__(self, data, current_time: datetime.datetime, end_time: datetime.datetime, current_battery,
                 target_battery,
                 interval: datetime.timedelta, action_set, starting_max_demand, pricing_model: pricing.PricingModel):

        super(EVA, self).__init__(data, current_time, end_time, current_battery, target_battery, interval, action_set,
                                  starting_max_demand, pricing_model)

        self.interval_in_minutes = self.interval.total_seconds() // 60

        self.g = nx.DiGraph()

        self.root = Node(current_battery, 0)

        charging_length = (end_time - current_time).total_seconds() // 60  # in minutes

        self.target = Node(target_battery, charging_length)

        self.billing_period = datetime.timedelta(days=30)

        self.g.add_node(self.root, usage_cost=np.inf, best_action=None, max_demand=np.inf)
        self.g.add_node(self.target, usage_cost=0, best_action=None, max_demand=starting_max_demand)

        self.prod_prediction = IER(self._data, current_time).predict([prod_algkey])  # todo check renes predictions plz
        self.cons_prediction = IEC(self._data[:current_time]).predict([cons_algkey])
        self.prod_prediction.index.tz_convert(dataset_tz)

    def get_real_time(self, node_time):
        return self.current_time + datetime.timedelta(minutes=node_time)

    def calc_usage_cost(self, time, interval: datetime.timedelta, ev_charge):
        """
        Calculates expected cost of house, ev, ier
        :param time: when we start charging
        :param interval: for how long we charge
        :param ev_charge: how much we charge the ev
        :return:
        """

        interval_in_minutes = interval.total_seconds() // 60
        if interval_in_minutes <= 0:
            return 0

        m2 = self.prod_prediction[time:time + interval - datetime.timedelta(minutes=1)][prod_algkey]
        m1 = self.cons_prediction[time:time + interval - datetime.timedelta(minutes=1)][cons_algkey]

        usage = m1 + ev_charge / interval_in_minutes - m2

        return self._pricing_model.get_usage_cost(usage)

    def calc_max_demand(self, time, interval, ev_charge):

        interval_in_minutes = interval.total_seconds() // 60

        # optimization for no-max demand pricing model since we don't actually care if there are no demand prices
        if interval_in_minutes <= 0 or not self._pricing_model.has_demand_prices():
            return 0

        demand = 60 * (
            self.cons_prediction[time:time + interval - datetime.timedelta(minutes=1)][cons_algkey] -
            self.prod_prediction[time:time + interval - datetime.timedelta(minutes=1)][
                prod_algkey])
        demand += 60 * (ev_charge / interval_in_minutes)
        # return max(demand.max(), 0)
        if interval_in_minutes == 15:
            return max(demand.mean(), 0)
        else:
            return max(demand.resample(datetime.timedelta(minutes=15)).mean().max(), 0)

    def shortest_path(self, from_node):
        """
        Creates our graph using DFS while we determine the best path
        :param from_node: the node we are currently on
        """

        # target.time is the time that the battery must be charged and from_node.time is the current time
        if from_node.time >= self.target.time:
            if from_node.battery < self.target.battery:
                # this (end) node is acceptable only if we have enough charge in the battery
                self.g.add_node(from_node, usage_cost=np.inf, best_action=None, max_demand=np.inf)
            return

        if from_node.battery >= self.target.battery:
            action_set = [0]
        else:
            action_set = self.action_set
        # shuffle(action_set)  # by shuffling we can achieve better pruning

        for action in action_set:
            new_battery, battery_consumption = calc_charge(action, self.interval, from_node.battery)

            new_node = Node(
                battery=new_battery,
                time=from_node.time + self.interval_in_minutes
            )

            # there are many instances where we can prune this new node
            # 1. if there's no time left to charge..
            charge_length = datetime.timedelta(minutes=self.target.time - new_node.time)
            max_battery, _ = calc_charge(max(self.action_set), charge_length, new_node.battery)
            if max_battery < self.target.battery:
                continue  # skip

            # calculate this path usage cost and demand
            interval_usage_cost = self.calc_usage_cost(self.get_real_time(from_node.time), self.interval,
                                                       battery_consumption)
            interval_demand = self.calc_max_demand(self.get_real_time(from_node.time), self.interval,
                                                   battery_consumption)

            demand_balancer = ((self.target.time - from_node.time) / (self.billing_period.total_seconds() / 60))
            # demand_balancer = datetime.timedelta(days=self.current_time.day) / datetime.timedelta(days=30)

            remaining_time = datetime.timedelta(minutes=self.target.time - new_node.time)
            ideal_demand_cost = self._pricing_model.get_demand_cost(
                demand_balancer * max(interval_demand, self.calc_max_demand(self.get_real_time(new_node.time),
                                                                            remaining_time, 0))
            )

            # 2. (continue pruning) if we are guaranteed to generate a more expensive path
            ideal_remaining_cost = (
                self.calc_usage_cost(self.get_real_time(new_node.time), remaining_time, 0)
                + interval_usage_cost  #
                + self._pricing_model.ideal_charging_cost(self.target.battery - new_node.battery)
                + ideal_demand_cost  # ideal demand cost from now on
            )

            if self.g.node[from_node]['usage_cost'] + self._pricing_model.get_demand_cost(
                            demand_balancer * self.g.node[from_node]['max_demand']) < ideal_remaining_cost:
                continue

            if new_node not in self.g:
                self.g.add_node(new_node, usage_cost=np.inf, best_action=None, max_demand=np.inf)
                self.shortest_path(new_node)

            this_path_usage_cost = self.g.node[new_node]['usage_cost'] + interval_usage_cost
            this_path_demand = max(self.g.node[new_node]['max_demand'], interval_demand)
            this_path_demand_cost = self._pricing_model.get_demand_cost(demand_balancer * this_path_demand)

            this_path_cost = this_path_usage_cost + this_path_demand_cost

            self.g.add_edge(from_node,
                            new_node,
                            action=action
                            )

            if this_path_cost < self.g.node[from_node]['usage_cost'] + self._pricing_model.get_demand_cost(
                            demand_balancer * self.g.node[from_node]['max_demand']):
                # replace the costs of the current node
                self.g.add_node(from_node,
                                best_action=new_node,
                                usage_cost=this_path_usage_cost,
                                max_demand=this_path_demand
                                )

    def reconstruct_path(self):
        cur = self.root
        path = [cur]

        while self.g.node[cur]['best_action'] is not None:
            cur = self.g.node[cur]['best_action']
            path.append(cur)

        return path

    def advise(self):
        self.shortest_path(self.root)
        path = self.reconstruct_path()
        result = pd.Series(0, index=self.result_index)

        result[:] = [self.g[path[n]][path[n + 1]]['action'] for n in range(len(path) - 1)]
        return result


class SimpleEVPlanner(EVPlanner):
    def __init__(self, data, current_time, end_time, current_battery, target_battery, interval, action_set,
                 starting_max_demand, pricing_model: pricing.PricingModel, is_informed, is_delayed,
                 delayed_start):

        super(SimpleEVPlanner, self).__init__(data, current_time, end_time, current_battery, target_battery, interval,
                                              action_set, starting_max_demand, pricing_model)

        self._is_informed = is_informed
        self._is_delayed = is_delayed

        if current_time.time() > datetime.time(12):  # if we are after noon
            cur_date = current_time.date()
        else:
            cur_date = current_time.date() - datetime.timedelta(days=1)  # else we are probably after midnight

        if is_delayed:
            self.delayed_start_time = dataset_tz.localize(
                datetime.datetime.combine(cur_date, self._pricing_model._cheap_period[0]))

    def calc_informed_charge(self):
        informed_charge = None

        for action in self.action_set:
            # we should calculate how long we have to charge.
            # in case of DELAYED we have from max(now, delayed_start) until morning
            if self._is_delayed:
                remaining_charge_time = self.end_time - max(self.current_time, self.delayed_start_time)
            else:
                remaining_charge_time = self.end_time - self.current_time

            max_battery, _ = calc_charge(action, remaining_charge_time, self.current_battery)
            if max_battery >= 1:  # we had enough time to fully charge the battery, so we select this value
                informed_charge = action
                break

        return informed_charge

    def advise(self):

        result = pd.Series(0, index=self.result_index)

        if self._is_informed:
            charge_rate = self.calc_informed_charge()
        else:
            charge_rate = 1

        if self._is_delayed:
            result[self.result_index >= self.delayed_start_time] = charge_rate
        else:
            result[:] = charge_rate

        return result


class SimpleEVPlannerDelayed(SimpleEVPlanner):
    _name = 'Simple-Delayed'

    def __init__(self, data, current_time, end_time, current_battery, target_battery, interval, action_set,
                 starting_max_demand, pricing_model: pricing.PricingModel):
        super().__init__(data, current_time, end_time, current_battery, target_battery,
                         interval, action_set, starting_max_demand, pricing_model,
                         is_informed=False, is_delayed=True, delayed_start=None)


class SimpleEVPlannerInformed(SimpleEVPlanner):
    _name = 'Simple-Informed'

    def __init__(self, data, current_time, end_time, current_battery, target_battery, interval, action_set,
                 starting_max_demand, pricing_model: pricing.PricingModel):
        super().__init__(data, current_time, end_time, current_battery, target_battery,
                         interval, action_set, starting_max_demand, pricing_model,
                         is_informed=True, is_delayed=False, delayed_start=None)


class SimpleEVPlannerDelayedInformed(SimpleEVPlanner):
    _name = 'Informed-Delayed'

    def __init__(self, data, current_time, end_time, current_battery, target_battery, interval, action_set,
                 starting_max_demand, pricing_model: pricing.PricingModel):
        super().__init__(data, current_time, end_time, current_battery,
                         target_battery, interval,
                         action_set, starting_max_demand, pricing_model,
                         is_informed=True, is_delayed=True, delayed_start=None)


class ChargingController:
    def __init__(self, data, agent_class, start: datetime.datetime, end: datetime.datetime,
                 pricing_model: pricing.PricingModel, max_demand=0, starting_charge=0.1,
                 active_MPC=True):
        self.data = data
        self._agent_class = agent_class
        self.start = start
        self.end = end
        self.max_starting_demand = max_demand
        self.starting_charge = starting_charge
        self.active_MPC = active_MPC

        self._pricing_model = pricing_model

        self.current_day = self.data[self.start:self.end][[real_production_key, real_consumption_key]]
        self.current_day['House'] = self.current_day[real_consumption_key]
        self.current_day['IER'] = self.current_day[real_production_key]
        self.current_day['EV'] = 0

    def calc_real_usage(self, time, interval, ev_charge):

        interval_in_minutes = interval.total_seconds() / 60
        if interval_in_minutes <= 0:
            return 0

        m2 = self.data[time:time + interval - datetime.timedelta(minutes=1)][real_production_key]
        m1 = self.data[time:time + interval - datetime.timedelta(minutes=1)][real_consumption_key]

        usage = m1 + ev_charge / interval_in_minutes - m2

        return self._pricing_model.get_usage_cost(usage)

    def calc_real_demand(self, time, interval: datetime.timedelta, ev_charge):
        interval_in_minutes = interval.total_seconds() / 60
        if interval_in_minutes <= 0 or not self._pricing_model.has_demand_prices():
            return 0

        demand = 60 * (
            self.data[time:time + interval - datetime.timedelta(minutes=1)][real_consumption_key] -
            self.data[time:time + interval - datetime.timedelta(minutes=1)][real_production_key])
        demand += 60 * (ev_charge / interval_in_minutes)

        if interval_in_minutes == 15:
            return max(demand.mean(), 0)
        else:
            return max(demand.resample(datetime.timedelta(minutes=15)).mean().max(), 0)

    # return max(demand.max(), 0)


    def run(self):

        interval = datetime.timedelta(minutes=15)
        max_depth = int((self.end - self.start).total_seconds() / (60 * 15))

        action_set = Charger.action_set

        target_charge = 1

        current_charge = self.starting_charge
        current_time = self.start

        usage_cost = 0
        max_demand = self.max_starting_demand

        robustness = []

        hide_tqdm = (not self.active_MPC) or suppress_tqdm

        for d in tqdm(range(max_depth), leave=False, disable=hide_tqdm):
            if self.active_MPC or 'advise_unit' not in locals():
                advise_unit = self._agent_class(
                    data=self.data,
                    current_time=current_time,
                    end_time=self.end,
                    current_battery=current_charge,
                    target_battery=target_charge,
                    interval=interval,
                    action_set=action_set,
                    starting_max_demand=max_demand,
                    pricing_model=self._pricing_model
                )

            if self.active_MPC:
                result = advise_unit.advise()
                action = result[0]
            else:  # if we are not using an mpc
                try:
                    action = result[d]
                except NameError:
                    result = advise_unit.advise()  # run it once and then just take values
                    action = result[d]

            # print(current_charge)
            current_charge, battery_consumption = calc_charge_with_error(action, interval, current_charge)
            # print(current_charge)
            # print("For time {} to {}, took action {} and charged to: {}".format(advise_unit.get_real_time(0), advise_unit.get_real_time(interval), action, current_charge))

            self.current_day.loc[
                self.current_day.index >= current_time, 'EV'] = battery_consumption * 60 / interval.total_seconds()

            # Taking said action
            usage_cost += self.calc_real_usage(current_time, interval, battery_consumption)
            max_demand = max(max_demand, self.calc_real_demand(current_time, interval, battery_consumption))

            current_time = current_time + interval

        robustness.append(current_charge)

        return usage_cost, max_demand, robustness


class BillingPeriodSimulator:
    def __init__(self, data, agent_class, pricing_model: pricing.PricingModel, month: datetime.date, use_mpc):

        self._data = data

        self._agent_class = agent_class

        self.pricing_model = pricing_model

        date_start = month.replace(day=1)  # first day of the month
        date_end = datetime.date(month.year, month.month + 1, 1) - datetime.timedelta(days=1)  # last day of the month

        self.use_mpc = use_mpc

        self.online_periods, self.offline_periods = self.generate_arrive_leave_times(date_start, date_end, dataset_tz)

        self.usage_cost = 0
        self.max_demand = 0
        self.robustness_list = []

        self.billing_period = pd.DataFrame(
            index=pd.date_range(
                datetime.datetime.combine(date_start, datetime.time()),
                datetime.datetime.combine(date_end, datetime.time()) + datetime.timedelta(days=1) - datetime.timedelta(
                    minutes=1),
                freq='T',
                tz=dataset_tz
            ),
            columns=['House', 'IER', 'EV']
        )

    def run(self):
        for online_period, offline_period in tqdm(zip_longest(self.online_periods, self.offline_periods),
                                                  total=len(self.online_periods), leave=True, disable=suppress_tqdm):
            # print("Running from {} to {}. Starting SoC: {}".format(t[0], t[1], t[2]))

            # first the offline period:
            mpc = ChargingController(self._data, self._agent_class, offline_period[0], offline_period[1],
                                     self.pricing_model,
                                     max_demand=self.max_demand,
                                     starting_charge=1)  # starting_charge == 1 means we will not charge! :-)
            unplugged_demand = mpc.calc_real_demand(mpc.start, mpc.end - mpc.start, 0)
            unplugged_usage = mpc.calc_real_usage(mpc.start, mpc.end - mpc.start, 0)

            self.billing_period.loc[mpc.current_day.index, 'House'] = mpc.current_day['House']
            self.billing_period.loc[mpc.current_day.index, 'IER'] = mpc.current_day['IER']
            self.billing_period.loc[mpc.current_day.index, 'EV'] = mpc.current_day['EV']

            self.max_demand = max(self.max_demand, unplugged_demand)
            self.usage_cost += unplugged_usage

            # then our online period!
            if online_period == None:
                continue  # offline periods are more since they are both in the beginning and at the end

            mpc = ChargingController(self._data, self._agent_class, online_period[0], online_period[1],
                                     self.pricing_model,
                                     max_demand=self.max_demand,
                                     starting_charge=online_period[2],
                                     active_MPC=self.use_mpc)
            day_usage_cost, day_max_demand, robustness = mpc.run()
            self.robustness_list.append(robustness)

            self.billing_period.loc[mpc.current_day.index, 'House'] = mpc.current_day['House']
            self.billing_period.loc[mpc.current_day.index, 'IER'] = mpc.current_day['IER']
            self.billing_period.loc[mpc.current_day.index, 'EV'] = mpc.current_day['EV']

            self.max_demand = max(self.max_demand, day_max_demand)
            self.usage_cost += day_usage_cost

    def draw_period(self):

        # ugly plotly fix:
        naive_df = self.billing_period.tz_localize(None)
        data = [
            go.Scatter(
                x=naive_df.index,  # assign x as the dataframe column 'x'
                y=naive_df['House'] - naive_df['IER'],
                name='Total house consumption (House - IER)'
            ),
            go.Scatter(
                x=naive_df.index,  # assign x as the dataframe column 'x'
                y=naive_df['EV'],
                name='EV Consumption'
            ),
        ]
        layout = go.Layout(
            title='Plot Title',
            xaxis=dict(
                title='Time (Local Time)',
            ),
            yaxis=dict(
                title='kWh',
            )
        )
        fig = go.Figure(data=data, layout=layout)
        py.plot(fig)

    def get_metadata(self):
        metadata = {
            'pricing': self.pricing_model._name,
            'month': self.online_periods[0][0].strftime("%b%Y"),
            'agent': self._agent_class._name,
            'mpc': self.use_mpc
        }
        return metadata

    def print_description(self):
        print('Pricing Model: {}'.format(self.pricing_model._name))
        print('Month: {}'.format(self.online_periods[0][0].strftime("%B %Y")))
        print('Agent: {}'.format(self._agent_class._name))
        print('Using MPC: {}'.format(self.use_mpc))

    def print_results(self):

        print("Robustness: {:0.2f}%".format(100 * np.mean(self.robustness_list)))
        print("Usage Cost: {:0.2f}$".format(self.usage_cost))
        print("Demand Cost: {:0.2f}$".format(self.pricing_model.get_demand_cost(self.max_demand)))
        print("Final Cost: {:0.2f}$".format(self.usage_cost + self.pricing_model.get_demand_cost(self.max_demand)))

    def generate_arrive_leave_times(self, start_date, end_date, tz):

        current_date = start_date

        prev_disconnect = datetime.datetime.combine(start_date, datetime.time())
        online_periods = []
        offline_periods = []

        while current_date < end_date:
            # generate a disconnect time around 7:40 in the morning
            round_interval = datetime.timedelta(minutes=15)
            disconnect_time = round_time(
                random_time(datetime.time(hour=7, minute=40), datetime.timedelta(minutes=34.2),
                            current_date + datetime.timedelta(days=1)),
                round_interval
            )
            connect_time = round_time(
                random_time(datetime.time(hour=18, minute=38), datetime.timedelta(minutes=53.4), current_date),
                round_interval
            )

            charging_timespan = disconnect_time - connect_time

            soc = np.random.uniform(0.5, 0.9)

            # quality control
            if (charging_timespan > datetime.timedelta(hours=15) or charging_timespan < datetime.timedelta(hours=8)):
                continue

            online_periods.append((tz.localize(connect_time), tz.localize(disconnect_time), soc))
            offline_periods.append((tz.localize(prev_disconnect), tz.localize(connect_time)))
            prev_disconnect = disconnect_time

            current_date += datetime.timedelta(days=1)

        # at the end just append a last disconnect time
        offline_periods.append((tz.localize(prev_disconnect),
                                tz.localize(datetime.datetime.combine(end_date, datetime.time(hour=23, minute=59)))))
        # print(*map(lambda x: "{} - {} : {:.2f}".format(str(x[0].time()), str(x[1].time()), x[2]), online_periods), sep='\n')
        # print(*map(lambda x: "{} - {}".format(str(x[0].time()), str(x[1].time())), offline_periods), sep='\n')

        return online_periods, offline_periods


def write_hdf(f, key, df, meta=None, complib='zlib'):
    """Append pandas dataframe to hdf5.

    Args:
    f       -- File path
    key     -- Store key
    df      -- Pandas dataframe
    complib -- Compress lib 

    NOTE: We use maximum compression w/ zlib.
    """

    with SafeHDF5Store(f, complevel=9, complib=complib) as store:
        df.to_hdf(store, key, format='table')
        store.get_storer(key).attrs.meta = meta


def main():
    # argument parser

    parser = argparse.ArgumentParser()

    location = parser.add_mutually_exclusive_group(required=True)
    location.add_argument('--us', action='store_true')
    location.add_argument('--uk', action='store_true')

    agent = parser.add_mutually_exclusive_group(required=True)
    agent.add_argument('--smartcharge', action='store_true')
    agent.add_argument('--informed', action='store_true')
    agent.add_argument('--delayed', action='store_true')
    agent.add_argument('--informed-delayed', action='store_true')

    parser.add_argument('--no-mpc', action='store_true')

    parser.add_argument('--suppress-tqdm', action='store_true')

    parser.add_argument('--month-index', metavar='i', type=int, nargs=1)

    args = parser.parse_args()

    cfg_filenames = ['config/common.yml']

    cfg = {}

    for f in cfg_filenames:
        with open(f, 'r') as ymlfile:
            cfg.update(yaml.load(ymlfile))


    # wind dataset is a multli index since it also has predictions from meteo stations
    wind_data = pd.read_csv("windpower.csv.gz", index_col=[0, 1], parse_dates=True)
    wind_data.index = wind_data.index.set_levels(
        [wind_data.index.levels[0], pd.to_timedelta(wind_data.index.levels[1])])
    wind_data = wind_data.tz_localize('UTC', level=0).tz_convert(dataset_tz, level=0)
    # simulate a billing period

    dataset = pd.read_csv('house_data.csv.gz', parse_dates=[0], index_col=0).tz_localize('UTC').tz_convert(dataset_tz)

    dataset['House Consumption'] = dataset['House Consumption'] * cfg['adjustment']['house-scale']
    dataset['WTG Production'] = dataset['WTG Production'] * cfg['adjustment']['wtg-scale']

    # in the dataset find valid months
    months_house = [x.date() for x in dataset.groupby(pd.TimeGrouper(freq='M')).count().index.tolist()]
    months_wind = [x.date() for x in wind_data.xs(datetime.timedelta(0), level=1).groupby(
        pd.TimeGrouper(freq='M')).count().index.tolist()]

    valid_months = set(months_house) & set(months_wind)
    if (set(months_house) != set(months_wind)):
        print("Warning: valid months in the two datasets were not 100% matching. Using months common to both.")

    print('Found the following {} valid months in the datasets: '.format(len(valid_months)))
    print(', '.join(map(str, [x.strftime("%B %Y") for x in sorted(valid_months)])))

    if args.uk:
        print("Using UK Pricing.")
        pricing_model = pricing.EuropePricingModel(dataset.index)
    else:
        print("Using US Pricing.")
        pricing_model = pricing.USPricingModel(dataset.index)

    month = sorted(list(valid_months))[args.month_index[0]]
    print("Running for month: {}".format(month.strftime("%B %Y")))

    agent = None
    if args.smartcharge:
        agent = EVA
    elif args.informed:
        agent = SimpleEVPlannerInformed
    elif args.delayed:
        agent = SimpleEVPlannerDelayed
    elif args.informed_delayed:
        agent = SimpleEVPlannerDelayedInformed

    use_mpc = not args.no_mpc

    if use_mpc:
        print("Using MPC")
    else:
        print("Not using MPC")

    global suppress_tqdm
    suppress_tqdm = args.suppress_tqdm

    # dont forget to seed our RNG!
    random_seed = cfg['random-seed'] + month.year + month.month
    np.random.seed(random_seed)
    random.seed(random_seed)

    simulator = BillingPeriodSimulator(dataset, agent, pricing_model, month, use_mpc)
    simulator.run()
    # and print results
    simulator.print_description()
    simulator.print_results()
    # simulator.draw_period()
    meta = simulator.get_metadata()
    # some extra metadata
    meta['wtg-scale'] = cfg['adjustment']['wtg-scale']
    meta['house-scale'] = cfg['adjustment']['house-scale']
    meta['actions'] = cfg['actions']
    meta['seed'] = random_seed
    meta['execution-date'] = datetime.datetime.now()

    name = "{}.{}.{}.{}".format(meta['agent'], meta['pricing'], meta['month'], meta['mpc'])
    # simulator.billing_period.tz_convert('UTC').to_csv('r1.csv.gz', compression='gzip')
    results = simulator.billing_period.tz_convert('UTC')
    cols = ['House', 'IER', 'EV']
    results[cols] = results[cols].apply(pd.to_numeric)

    write_hdf('results.h5', name, results, meta=meta, complib='zlib')


if __name__ == '__main__':
    # some constants (column names)
    with open("config/common.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    # prod_algkey = cfg['algorithms']['production']
    prod_algkey = 'Renes'

    prod_algkey_var = cfg['algorithms']['production_var']
    cons_algkey = cfg['algorithms']['consumption']

    real_consumption_key = cfg['truth']['consumption']
    real_production_key = cfg['truth']['production']

    dataset_tz = pytz.timezone(cfg['dataset']['timezone'])
    del cfg

    main()

"""
EV advise unit
"""
import argparse
import datetime
import random

import networkx as nx
import numpy as np
import pandas as pd
import pytz
import yaml
from tqdm import tqdm

import pricing
from battery import Charger as ChargerClass
from house import IEC
from ier.ier import IER

# a global charger to take advantage of result caching
Charger = ChargerClass()


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

        m2 = self.prod_prediction[time:time + interval][prod_algkey]
        m1 = self.cons_prediction[time:time + interval][cons_algkey]

        usage = m1 + ev_charge / interval_in_minutes - m2

        return self._pricing_model.get_usage_cost(usage)

    def calc_max_demand(self, time, interval, ev_charge):

        interval_in_minutes = interval.total_seconds() // 60

        # optimization for no-max demand pricing model since we don't actually care if there are no demand prices
        if interval_in_minutes <= 0 or not self._pricing_model.has_demand_prices():
            return 0

        demand = 60 * (
            self.cons_prediction[time:time + interval][cons_algkey] - self.prod_prediction[time:time + interval][
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
            self.delayed_start_time = dataset_tz.localize(datetime.datetime.combine(cur_date, delayed_start))

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
    def __init__(self, data, current_time, end_time, current_battery, target_battery, interval, action_set,
                 starting_max_demand, pricing_model: pricing.PricingModel):
        super().__init__(data, current_time, end_time, current_battery, target_battery,
                         interval, action_set, starting_max_demand, pricing_model,
                         is_informed=False, is_delayed=True, delayed_start=None)


class SimpleEVPlannerInformed(SimpleEVPlanner):
    def __init__(self, data, current_time, end_time, current_battery, target_battery, interval, action_set,
                 starting_max_demand, pricing_model: pricing.PricingModel):
        super().__init__(data, current_time, end_time, current_battery, target_battery,
                         interval, action_set, starting_max_demand, pricing_model,
                         is_informed=True, is_delayed=False, delayed_start=None)


class SimpleEVPlannerDelayedInformed(SimpleEVPlanner):
    def __init__(self, data, current_time, end_time, current_battery, target_battery, interval, action_set,
                 starting_max_demand, pricing_model: pricing.PricingModel):
        super().__init__(data, current_time, end_time, current_battery,
                         target_battery, interval,
                         action_set, starting_max_demand, pricing_model,
                         is_informed=True, is_delayed=True, delayed_start=None)


class DaySimulator:
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

    def calc_real_usage(self, time, interval, ev_charge):

        interval_in_minutes = interval.total_seconds() / 60
        if interval_in_minutes <= 0:
            return 0

        m2 = self.data[time:time + interval][real_production_key]
        m1 = self.data[time:time + interval][real_consumption_key]

        usage = m1 + ev_charge / interval_in_minutes - m2

        return self._pricing_model.get_usage_cost(usage)

    def calc_real_demand(self, time, interval: datetime.timedelta, ev_charge):
        interval_in_minutes = interval.total_seconds() / 60
        if interval_in_minutes <= 0 or not self._pricing_model.has_demand_prices():
            return 0

        demand = 60 * (
            self.data[time:time + interval][real_consumption_key] - self.data[time:time + interval][
                real_production_key])
        demand += 60 * (ev_charge / interval_in_minutes)

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

        current_day = self.data[self.start:self.end][[real_production_key, real_consumption_key]]
        current_day['House'] = current_day[real_production_key] - current_day[real_consumption_key]
        current_day['EV'] = 0

        robustness = []

        for d in tqdm(range(max_depth), leave=False):
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
                # advise_unit = SimpleEVPlanner(
                #     data=self.data,
                #     current_time=current_time,
                #     end_time=self.end,
                #     current_battery=current_charge,
                #     target_battery=target_charge,
                #     interval=interval,
                #     action_set=action_set,
                #     starting_max_demand=max_demand,
                #     pricing_model=self._pricing_model,
                #     is_informed=(self._cfg['advise-unit'] in ['informed', 'informed-delayed']),
                #     is_delayed=(self._cfg['advise-unit'] in ['delayed', 'informed-delayed']),
                #     delayed_start=datetime.time(hour=self._cfg['delay']['hour'],
                #                                 minute=self._cfg['delay']['minute']) if self._cfg[
                #                                                                             'advise-unit'] in [
                #                                                                             'delayed',
                #                                                                             'informed-delayed'] else None
                # )

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

            current_day.loc[
                current_day.index >= current_time, 'EV'] = battery_consumption * 60 / interval.total_seconds()

            # Taking said action

            usage_cost += self.calc_real_usage(current_time, interval, battery_consumption)
            max_demand = max(max_demand, self.calc_real_demand(current_time, interval, battery_consumption))

            current_time = current_time + interval

        robustness.append(current_charge)

        # print("Battery cache hit: {:0.2f}".format(Charger.hit / Charger.total))
        # print("Battery State: {:0.2f}%".format(100 * current_charge))
        # print("Usage Cost: {:0.2f}$".format(usage_cost))
        # print("Demand Cost: {:0.2f}$".format(self._pricing_model.get_demand_cost(max_demand)))
        # print("Final Cost: {:0.2f}$".format(usage_cost + self._pricing_model.get_demand_cost(max_demand)))
        # import plotly.offline as py
        # import plotly.graph_objs as go
        # data = [
        #     go.Scatter(
        #         x=current_day.index,  # assign x as the dataframe column 'x'
        #         y=current_day['House'],
        #         name='House'
        #     ),
        #     go.Scatter(
        #         x=current_day.index,  # assign x as the dataframe column 'x'
        #         y=-current_day['EV'] + current_day['House'],
        #         name='House & EV'
        #     ),
        # ]
        # py.plot(data)


        return usage_cost, max_demand, robustness


class BillingPeriodSimulator:
    def __init__(self, data, agent_class, pricing_model: pricing.PricingModel, month: datetime.date):

        self._data = data

        self._agent_class = agent_class

        self.pricing_model = pricing_model

        date_start = month.replace(day=1)  # first day of the month
        date_end = datetime.date(month.year, month.month + 1, 1) - datetime.timedelta(days=1)  # last day of the month

        self.test_times = self.generate_arrive_leave_times(date_start, date_end, dataset_tz)

        self.usage_cost = 0
        self.max_demand = 0
        self.robustness_list = []

    def run(self):

        for index, t in tqdm(enumerate(self.test_times), total=len(self.test_times), leave=True):
            # print("Running from {} to {}. Starting SoC: {}".format(t[0], t[1], t[2]))

            # print("Always using MPC")
            use_mpc = True

            mpc = DaySimulator(self._data, self._agent_class, t[0], t[1], self.pricing_model,
                               max_demand=self.max_demand,
                               starting_charge=t[2],
                               active_MPC=use_mpc)
            day_usage_cost, day_max_demand, robustness = mpc.run()

            self.robustness_list.append(robustness)

            self.max_demand = max(self.max_demand, day_max_demand)

            # creating a 'fake' mpc for the inbetween hours

            try:
                mpc = DaySimulator(self._data, self._agent_class, t[1], self.test_times[index + 1][0],
                                   self.pricing_model,
                                   max_demand=self.max_demand,
                                   starting_charge=1)
                unplugged_demand = mpc.calc_real_demand(mpc.start, mpc.end - mpc.start, 0)
                unplugged_usage = mpc.calc_real_usage(mpc.start, mpc.end - mpc.start, 0)

                # print("========", unplugged_demand, max_demand, "===========")

                self.max_demand = max(self.max_demand, unplugged_demand)
                self.usage_cost += unplugged_usage
            except IndexError:
                # our period is over :-)
                pass

            self.usage_cost += day_usage_cost

    def print_description(self):
        print('Location: {}'.format(self._cfg['location']))
        print('Month: {}'.format(self._cfg['dates']['month']))
        print('Agent: {}'.format(self._cfg['advise-unit']))
        print('Using MPC: {}'.format(self._cfg['USE_MPC']))
        if self._cfg['advise-unit'] in ['delayed', 'informed-delayed']:
            print('Delayed Start: {}:{}'.format(self._cfg['delay']['hour'], self._cfg['delay']['minute']))

    def print_results(self):

        print("Robustness: {:0.2f}%".format(100 * np.mean(self.robustness_list)))
        print("Usage Cost: {:0.2f}$".format(self.usage_cost))
        print("Demand Cost: {:0.2f}$".format(self.pricing_model.get_demand_cost(self.max_demand)))
        print("Final Cost: {:0.2f}$".format(self.usage_cost + self.pricing_model.get_demand_cost(self.max_demand)))

    def generate_arrive_leave_times(self, start_date, end_date, tz):

        current_date = start_date
        time_list = []

        while current_date < end_date:

            current_datetime = datetime.datetime.combine(current_date, datetime.time())

            # mean/std as taken from: Stochastic Optimal Energy Management of Smart Home with PEV Energy Storage

            end_minutes = np.random.normal(460, 34.2)
            start_minutes = np.random.normal(1118, 53.4)

            # round to fifteen (minutes)
            base = 15
            round_to_fifteen = lambda x: int(base * round(float(x) / base))

            start_minutes = round_to_fifteen(start_minutes)
            end_minutes = round_to_fifteen(end_minutes)

            # creating datetime

            start_time = current_datetime + datetime.timedelta(minutes=start_minutes)
            end_time = current_datetime + datetime.timedelta(days=1, minutes=end_minutes)

            # putting timezone info

            start_time = tz.localize(start_time)
            end_time = tz.localize(end_time)

            soc = np.random.uniform(0.2, 0.8)

            # print("{} to {}".format(start_time, end_time))
            # print(end_time-start_time)

            # quality control
            if (end_time - start_time > datetime.timedelta(hours=15) or end_time - start_time < datetime.timedelta(
                    hours=8)):
                continue

            time_list.append((start_time, end_time, soc))
            current_date += datetime.timedelta(days=1)

        return time_list


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
    agent.add_argument('--informed_delayed', action='store_true')

    args = parser.parse_args()

    cfg_filenames = ['config/common.yml']

    cfg = {}

    for f in cfg_filenames:
        with open(f, 'r') as ymlfile:
            cfg.update(yaml.load(ymlfile))

    # dont forget to seed our RNG!
    np.random.seed(cfg['random-seed'])
    random.seed(cfg['random-seed'])

    # wind dataset is a multli index since it also has predictions from meteo stations
    wind_data = pd.read_csv("windpower.csv.gz", index_col=[0, 1], parse_dates=True)
    wind_data.index = wind_data.index.set_levels(
        [wind_data.index.levels[0], pd.to_timedelta(wind_data.index.levels[1])])

    # simulate a billing period

    dataset = pd.read_csv('house_data.csv.gz', parse_dates=[0], index_col=0).tz_localize('UTC').tz_convert(dataset_tz)

    # in the dataset find valid months
    months_house = [x.date() for x in dataset.groupby(pd.TimeGrouper(freq='M')).count().index.tolist()]
    months_wind = [x.date() for x in wind_data.xs(datetime.timedelta(0), level=1).groupby(
        pd.TimeGrouper(freq='M')).count().index.tolist()]

    valid_months = set(months_house) & set(months_wind)
    if (set(months_house) != set(months_wind)):
        print("Warning: valid months in the two datasets were not 100% matching. Using months common to both.")

    print('Found the following {} valid months in the datasets: '.format(len(valid_months)))
    print(', '.join(map(str, [x.strftime("%B %Y") for x in valid_months])))

    if args.uk:
        print("Using UK Pricing.")
        pricing_model = pricing.EuropePricingModel(dataset.index)
    else:
        print("Using US Pricing.")
        pricing_model = pricing.USPricingModel(dataset.index)

    # choosing a valid month! (first one atm)
    month = sorted(list(valid_months))[6]
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

    simulator = BillingPeriodSimulator(dataset, agent, pricing_model, month)
    simulator.run()
    # and print results
    # simulator.print_description()
    simulator.print_results()


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

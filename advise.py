"""
EV advise unit
"""
import timeit
from random import shuffle
import random

import networkx as nx
import numpy as np
import plotly.offline as py
import plotly.graph_objs as go
import pandas as pd

from tqdm import tqdm

from house.iec import IEC
from ier.ier import IER
from utils.utils import plotly_figure

import datetime
import pytz

from battery.battery import Charger

europe = True

random.seed(1337)

historical_offset = 2000

prod_algkey = 'Renes Hybrid'
prod_algkey_var = 'Renes Hybrid STD'
cons_algkey = 'Baseline Finder'

real_consumption_key = 'House Consumption'
real_production_key = 'WTG Production'

# cons_algkey_var = 'Baseline Finder Hybrid STD'

dataset_filename = 'dataset-kw.gz'
dataset_tz = 'Europe/Zurich'

print("Reading dataset...", end='')
start_time = timeit.default_timer()
dataset = pd.read_csv(dataset_filename, parse_dates=[0], index_col=0).tz_localize('UTC').tz_convert(dataset_tz)
elapsed = timeit.default_timer() - start_time
print("done ({:0.2f}sec)".format(elapsed))

dataset = dataset[
          datetime.datetime(2012, 9, 7, 0, 0, 0):]  # trimming the dataset because before that we have bad values

# Create Usage Cost Column
usage_cost_key = 'Buy Price'
sell_price_key = 'Sell Price'

peak_day_pricing = True  # https://www.pge.com/en_US/business/rate-plans/rate-plans/peak-day-pricing/peak-day-pricing.page
print("Setting up prices...", end='')
start_time = timeit.default_timer()
if europe:
    # https://economy10.files.wordpress.com/2016/12/economy10-com-survey-results-dec-20163.pdf
    dataset[usage_cost_key] = 0.16  # peak energy

    # off peak hours: https://www.ovoenergy.com/guides/energy-guides/economy-10.html
    dataset.loc[(dataset.index.time > datetime.time(hour=13, minute=00)) & (
        dataset.index.time < datetime.time(hour=16, minute=00)), usage_cost_key] = 0.107

    dataset.loc[(dataset.index.time > datetime.time(hour=20, minute=00)) & (
        dataset.index.time < datetime.time(hour=22, minute=00)), usage_cost_key] = 0.107

    dataset.loc[(dataset.index.time > datetime.time(hour=00, minute=00)) & (
        dataset.index.time < datetime.time(hour=5, minute=00)), usage_cost_key] = 0.107

    # https://www.gov.uk/feed-in-tariffs/overview
    dataset[sell_price_key] = 0.0485
    min_price = 0.0485

elif peak_day_pricing:
    # summer only
    dataset[usage_cost_key] = 0.202
    dataset.loc[(dataset.index.time > datetime.time(hour=8, minute=30)) & (
        dataset.index.time < datetime.time(hour=21, minute=30)), usage_cost_key] = 0.230
    dataset.loc[(dataset.index.time > datetime.time(hour=12, minute=00)) & (
        dataset.index.time < datetime.time(hour=18, minute=00)), usage_cost_key] = 0.253

    dataset[usage_cost_key] = dataset[usage_cost_key]

    min_price = 0.202
else:
    raise (ValueError)

elapsed = timeit.default_timer() - start_time
print("done ({:0.2f}sec)".format(elapsed))

print(dataset.describe())


def calc_demand_cost(max_demand):
    return max_demand * 8.03




def calc_charge(action, interval, cur_charge):
    # Given that Charging rates are in kW and self.interval is in minutes, returns joules

    charger = Charger(cur_charge)

    return charger.charge(action, interval)


def calc_charge_with_error(action, interval, cur_charge):
    # Given that Charging rates are in kW and self.interval is in minutes, returns joules

    charger = Charger(cur_charge)

    current_charge, battery_consumption = charger.charge(action, interval)

    current_charge += np.random.normal(0, 0.05 * (current_charge - cur_charge) / 3)

    return current_charge, battery_consumption


def min_charging_cost(charge):
    return min_price * charge


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


class EVA:
    def __init__(self, current_time, target, interval, action_set, root=Node(0, 0), starting_max_demand=0):


        self.current_time = current_time


        self.g = nx.DiGraph()
        self.interval = interval
        self.action_set = action_set
        self.root = root
        self.target = target

        self.billing_period = 30 * 24 * 60 / interval  # 30 days

        self.g.add_node(root, usage_cost=np.inf, best_action=None, max_demand=np.inf)
        self.g.add_node(target, usage_cost=0, best_action=None, max_demand=starting_max_demand)

        self.prod_prediction = IER(dataset, current_time).predict([prod_algkey])  # todo check renes predictions plz
        self.cons_prediction = IEC(dataset[:current_time]).predict([cons_algkey])
        self.prod_prediction.index.tz_convert(pytz.timezone(dataset_tz))


    def get_real_time(self, node_time):
        return self.current_time + datetime.timedelta(minutes=node_time)

    def calc_usage_cost(self, time, interval, ev_charge):
        """
        Calculates expected cost of house, ev, ier
        :param time: when we start charging
        :param self.interval: for how long we charge
        :param ev_charge: how much we charge the ev
        :return:
        """
        if interval <= 0:
            return 0

        interval_in_minutes = interval

        interval = datetime.timedelta(minutes=interval)

        m2 = self.prod_prediction[time:time + interval][prod_algkey]
        m1 = self.cons_prediction[time:time + interval][cons_algkey]

        # usage = pd.DataFrame()
        #
        # usage['p'] = m2
        # usage['c'] = m1
        # usage['e'] = ev_charge / interval_in_minutes

        pbuy = dataset[time:time + interval][usage_cost_key]

        if europe:
            psell = dataset[time:time + interval][sell_price_key]
            usage = m1 + ev_charge / interval_in_minutes - m2
            price = pbuy.copy()
            price[usage < 0] = psell
            final = (usage * price).sum()

        else:
            final = ((m1 + ev_charge / interval_in_minutes - m2) * pbuy).sum()  #pbuy == psell

        return final

    def calc_max_demand(self, time, interval, ev_charge):

        if interval <= 0 or europe:
            return 0
        interval_in_minutes = interval

        # time = current_time + datetime.timedelta(minutes=time)
        interval = datetime.timedelta(minutes=interval)

        demand = 60 * (
        self.cons_prediction[time:time + interval][cons_algkey] - self.prod_prediction[time:time + interval][
            prod_algkey])
        demand += 60 * (ev_charge / interval_in_minutes)

        return max(demand.max(), 0)

    def shortest_path(self, from_node):
        """
        Creates our graph using DFS while we determine the best path
        :param from_node: the node we are currently on
        """

        # target.time is the time that the battery must be charged and from_node.time is the current time
        if from_node.time >= self.target.time:
            if from_node.battery < self.target.battery:
                # this (end) node is acceptable only if we have enough charge in the battery
                self.g.add_node(from_node, usage_cost=np.inf, demand_cost=np.inf, best_action=None, max_demand=np.inf)
            return

        if (from_node.battery >= self.target.battery):
            action_set = [0]
        else:
            action_set = self.action_set
            shuffle(action_set)  # by shuffling we can achieve better pruning

        for action in action_set:
            new_battery, battery_consumption = calc_charge(action, self.interval, from_node.battery)

            new_node = Node(
                battery=new_battery,
                time=from_node.time + self.interval
            )

            # there are many instances where we can prune this new node
            # 1. if there's no time left to charge..
            max_battery, _ = calc_charge(max(self.action_set), self.target.time - new_node.time, new_node.battery)
            if max_battery < self.target.battery:
                continue  # skip

            # calculate this path usage cost and demand
            interval_usage_cost = self.calc_usage_cost(self.get_real_time(from_node.time), self.interval,
                                                       battery_consumption)
            interval_demand = self.calc_max_demand(self.get_real_time(from_node.time), self.interval,
                                                   battery_consumption)

            demand_balancer = ((self.target.time - from_node.time) / self.billing_period)

            ideal_demand_cost = calc_demand_cost(
                demand_balancer * max(interval_demand, self.calc_max_demand(self.get_real_time(new_node.time),
                                                                            self.target.time - new_node.time, 0))
            )

            # 2. (continue pruning) if we are guaranteed to generate a more expensive path
            ideal_remaining_cost = (
                self.calc_usage_cost(self.get_real_time(new_node.time), self.target.time - new_node.time, 0)
                + interval_usage_cost  #
                + min_charging_cost(self.target.battery - new_node.battery)
                + ideal_demand_cost  # ideal demand cost from now on
            )

            if self.g.node[from_node]['usage_cost'] + calc_demand_cost(
                            demand_balancer * self.g.node[from_node]['max_demand']) < ideal_remaining_cost:
                continue

            if new_node not in self.g:
                self.g.add_node(new_node, demand_cost=np.inf, usage_cost=np.inf, best_action=None, max_demand=np.inf)
                self.shortest_path(new_node)

            this_path_usage_cost = self.g.node[new_node]['usage_cost'] + interval_usage_cost
            this_path_demand = max(self.g.node[new_node]['max_demand'], interval_demand)
            this_path_demand_cost = calc_demand_cost(demand_balancer * this_path_demand)

            this_path_cost = this_path_usage_cost + this_path_demand_cost

            self.g.add_edge(from_node,
                            new_node,
                            action=action
                            )

            if this_path_cost < self.g.node[from_node]['usage_cost'] + calc_demand_cost(
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


class MPC:
    def __init__(self, data, start, end, max_demand=0, starting_charge=0.1):
        self.data = data
        self.start = start
        self.end = end
        self.max_starting_demand = max_demand
        self.starting_charge = starting_charge

    def calc_real_usage(self, time, interval, ev_charge):
        if interval <= 0:
            return 0

        interval_in_minutes = interval

        interval = datetime.timedelta(minutes=interval)

        m2 = self.data[time:time + interval][real_production_key]
        m1 = self.data[time:time + interval][real_consumption_key]

        pbuy = dataset[time:time + interval][usage_cost_key]
        if europe:
            psell = dataset[time:time + interval][sell_price_key]
            usage = m1 + ev_charge / interval_in_minutes - m2
            price = pbuy.copy()
            price[usage < 0] = psell
            final = (usage * price).sum()

        else:
            final = ((m1 + ev_charge / interval_in_minutes - m2) * pbuy).sum()  # pbuy == psell

        return final

    def calc_real_demand(self, time, interval, ev_charge):
        if interval <= 0 or europe:
            return 0
        interval_in_minutes = interval

        # time = current_time + datetime.timedelta(minutes=time)
        interval = datetime.timedelta(minutes=interval)

        demand = 60 * (
        self.data[time:time + interval][real_consumption_key] - self.data[time:time + interval][real_production_key])
        demand += 60 * (ev_charge / interval_in_minutes)

        return max(demand.max(), 0)



    def run(self):

        dummy = False

        interval = 15
        max_depth = int((self.end - self.start).total_seconds() / (60 * interval))

        action_set = Charger(0).action_set

        target_charge = 1

        current_charge = self.starting_charge
        current_time = self.start

        usage_cost = 0
        max_demand = self.max_starting_demand

        current_day = self.data[self.start:self.end][[real_production_key, real_consumption_key]]
        current_day['House'] = current_day[real_production_key] - current_day[real_consumption_key]
        current_day['EV'] = 0

        for d in tqdm(range(max_depth, 0, -1)):
            root = Node(current_charge, 0)

            if not dummy:
                advise_unit = EVA(
                    current_time=current_time,
                    target=Node(target_charge, interval * d),
                    interval=interval,
                    action_set=action_set,
                    root=root,
                    starting_max_demand=max_demand
                )
                advise_unit.shortest_path(root)
                path = advise_unit.reconstruct_path()

                # fig = plotly_figure(advise_unit.g, path=path)
                # py.plot(fig)
                action = advise_unit.g[path[0]][path[1]]['action']


            else:
                if current_charge < 1:
                    action = 1
                else:
                    action = 0
            # print(current_charge)
            current_charge, battery_consumption = calc_charge_with_error(action, interval, current_charge)

            # print(current_charge)
            # print("For time {} to {}, took action {} and charged to: {}".format(advise_unit.get_real_time(0), advise_unit.get_real_time(interval), action, current_charge))

            current_day.loc[current_day.index >= current_time, 'EV'] = battery_consumption / interval

            # Taking said action

            usage_cost += self.calc_real_usage(current_time, interval, battery_consumption)
            max_demand = max(max_demand, self.calc_real_demand(current_time, interval, battery_consumption))

            current_time = current_time + datetime.timedelta(minutes=interval)

        print("Usage Cost: {:0.2f}$".format(usage_cost))
        print("Demand Cost: {:0.2f}$".format(calc_demand_cost(max_demand)))
        print("Final Cost: {:0.2f}$".format(usage_cost + calc_demand_cost(max_demand)))

        data = [
            go.Scatter(
                x=current_day.index,  # assign x as the dataframe column 'x'
                y=current_day['House'],
                name='House'
            ),
            go.Scatter(
                x=current_day.index,  # assign x as the dataframe column 'x'
                y=-current_day['EV'] + current_day['House'],
                name='House & EV'
            ),
        ]
        py.plot(data)

        return usage_cost, max_demand


def generate_arrive_leave_times(start_date, end_date):
    np.random.seed(1337)

    current_date = start_date

    timezone = pytz.timezone(dataset_tz)

    time_list = []

    while (current_date < end_date):

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
        start_time = start_time.replace(tzinfo=timezone)
        end_time = end_time.replace(tzinfo=timezone)

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


if __name__ == '__main__':
    time = dataset.index[-historical_offset]

    test_times = generate_arrive_leave_times(datetime.date(2013, 1, 1), datetime.date(2013, 1, 31))

    usage_cost = 0
    max_demand = 0

    for t in test_times:
        print("Running from {} to {}. Starting SoC: {}".format(t[0], t[1], t[2]))

        mpc = MPC(dataset, t[0], t[1], max_demand=max_demand, starting_charge=t[2])
        day_usage_cost, day_max_demand = mpc.run()

        max_demand = max(max_demand, day_max_demand)
        usage_cost += day_usage_cost

    print("Usage Cost: {:0.2f}$".format(usage_cost))
    print("Demand Cost: {:0.2f}$".format(calc_demand_cost(max_demand)))
    print("Final Cost: {:0.2f}$".format(usage_cost + calc_demand_cost(max_demand)))



    # interval = 15
    # max_depth = 64
    #
    # action_set = [0, 0.25, 0.5, 0.75, 1]
    # target_charge = 1
    #
    # target_time = max_depth * interval
    #
    #
    #
    # print("Calculating optimal path...", end='')
    # start_time = timeit.default_timer()
    # advise_unit = EVA(
    #     target=Node(target_charge, target_time),
    #     interval=interval,
    #     action_set=action_set,
    #     root=root
    # )
    #
    # advise_unit.shortest_path(from_node=root)
    #
    # elapsed = timeit.default_timer() - start_time
    # print("done ({:0.2f}sec)".format(elapsed))
    # path = advise_unit.reconstruct_path()
    # path_edges = list(zip(path, path[1:]))
    #
    # fig = plotly_figure(advise_unit.g, path=path)
    # py.plot(fig)
    #
    # #print(len(advise_unit.g.nodes()))
    # predictions = prod_prediction[prod_algkey] - cons_prediction[cons_algkey]
    # ground_truth = dataset[current_time:current_time+datetime.timedelta(hours=16)]['WTG Production'] - dataset[current_time:current_time+datetime.timedelta(hours=16)]['House Consumption']
    #
    # data = [
    #     go.Scatter(
    #         x=predictions.index,  # assign x as the dataframe column 'x'
    #         y=predictions
    #     ),
    #     go.Scatter(
    #         x=ground_truth.index,  # assign x as the dataframe column 'x'
    #         y=ground_truth
    #     )
    # ]
    # IPython notebook
    # py.iplot(data, filename='pandas/basic-line-plot')

    #py.plot(data)

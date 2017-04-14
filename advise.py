"""
EV advise unit
"""
import timeit
from random import shuffle

import networkx as nx
import numpy as np
import plotly.offline as py
import plotly.graph_objs as go
import pandas as pd

from house.iec import IEC
from ier.ier import IER
from utils.utils import plotly_figure

import datetime

from battery.battery import Charger

historical_offset = 2000

prod_algkey = 'Renes Hybrid'
prod_algkey_var = 'Renes Hybrid STD'
cons_algkey = 'Baseline Finder'

# cons_algkey_var = 'Baseline Finder Hybrid STD'

dataset_filename = 'dataset-kw.gz'
dataset_tz = 'Europe/Zurich'

print("Reading dataset...", end='')
start_time = timeit.default_timer()
dataset = pd.read_csv(dataset_filename, parse_dates=[0], index_col=0).tz_localize('UTC').tz_convert(dataset_tz)
elapsed = timeit.default_timer() - start_time
print("done ({:0.2f}sec)".format(elapsed))


# Create Usage Cost Column
usage_cost_key = 'Usage Cost'
peak_day_pricing = True  # https://www.pge.com/en_US/business/rate-plans/rate-plans/peak-day-pricing/peak-day-pricing.page
print("Setting up prices...", end='')
start_time = timeit.default_timer()
if peak_day_pricing:
    # summer only
    dataset[usage_cost_key] = 0.202
    dataset.loc[(dataset.index.time > datetime.time(hour=8, minute=30)) & (
    dataset.index.time < datetime.time(hour=21, minute=30)), usage_cost_key] = 0.230
    dataset.loc[(dataset.index.time > datetime.time(hour=12, minute=00)) & (
    dataset.index.time < datetime.time(hour=18, minute=00)), usage_cost_key] = 0.253

    dataset[usage_cost_key] = dataset[usage_cost_key] / 60

    min_price = 0.202
elapsed = timeit.default_timer() - start_time
print("done ({:0.2f}sec)".format(elapsed))




def calc_demand_cost(max_demand):
    return max_demand * 8.03




def calc_charge(action, interval, cur_charge):
    # Given that Charging rates are in kW and self.interval is in minutes, returns joules

    charger = Charger(cur_charge)

    return charger.charge(action, interval)


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

        self.cons_prediction = IEC(dataset[:current_time]).predict([cons_algkey])
        self.prod_prediction = IER(dataset, historical_offset).predict(
            [prod_algkey])  # todo check renes predictions plz

        self.g = nx.DiGraph()
        self.interval = interval
        self.action_set = action_set
        self.root = root
        self.target = target

        self.billing_period = 30 * 24 * 60 / interval  # 30 days

        self.g.add_node(root, usage_cost=np.inf, best_action=None, max_demand=np.inf)
        self.g.add_node(target, usage_cost=0, best_action=None, max_demand=starting_max_demand)

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

        usage = pd.DataFrame()

        usage['p'] = m2
        usage['c'] = m1
        usage['e'] = ev_charge / interval_in_minutes

        pbuy = dataset[time:time + interval][usage_cost_key]

        final = ((usage['c'] + usage['e'] - usage['p']) * pbuy).sum()

        return final

    def calc_max_demand(self, time, interval, ev_charge):
        if interval <= 0:
            return 0
        interval_in_minutes = interval

        # time = current_time + datetime.timedelta(minutes=time)
        interval = datetime.timedelta(minutes=interval)

        demand = self.cons_prediction[time:time + interval][cons_algkey] - self.prod_prediction[time:time + interval][
            prod_algkey]
        demand += ev_charge / interval_in_minutes

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

        shuffle(self.action_set)  # by shuffling we can achieve better pruning
        for action in self.action_set:
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

            ideal_demand_cost = calc_demand_cost(demand_balancer * max(interval_demand, self.calc_max_demand(
                self.get_real_time(new_node.time), self.target.time - new_node.time, 0)))

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
    def __init__(self, data, start, end):
        self.data = data
        self.start = start
        self.end = end

    def run(self):
        interval = 15
        max_depth = int((self.end - self.start).total_seconds() / (60 * interval))

        action_set = [0, 0.5, 1]
        target_charge = 1

        for d in range(max_depth, 1, -1):
            print("Running advise for depth: {}".format(d))
            root = Node(0.5, 0)

            advise_unit = EVA(
                current_time=self.start,
                target=Node(target_charge, interval * d),
                interval=interval,
                action_set=action_set,
                root=root
            )
            advise_unit.shortest_path(root)
            path = advise_unit.reconstruct_path()

            # fig = plotly_figure(advise_unit.g, path=path)
            #py.plot(fig)

            action = advise_unit.g[path[0]][path[1]]['action']
            print(action)


if __name__ == '__main__':
    time = dataset.index[-historical_offset]

    mpc = MPC(dataset, time, time + datetime.timedelta(hours=12))

    mpc.run()


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

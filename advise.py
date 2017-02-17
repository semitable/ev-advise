"""
EV advise unit
"""

import math
from math import sqrt

import networkx as nx
import numpy as np
import pandas as pd
import plotly.offline as py
from scipy.special import erf, erfc

from house.iec import IEC
from ier.ier import IER
from utils.utils import plotly_figure

buy_price = 0.00006
sell_price = 0.00005

historical_offset = 2000

prod_algkey = 'Renes Hybrid'
prod_algkey_var = 'Renes Hybrid STD'
cons_algkey = 'Baseline Finder Hybrid'
cons_algkey_var = 'Baseline Finder Hybrid STD'

dataset_filename = 'dataset.gz'
dataset_tz = 'Europe/Zurich'

dataset = pd.read_csv(dataset_filename, parse_dates=[0], index_col=0).tz_localize('UTC').tz_convert(dataset_tz)

current_time = dataset.index[-historical_offset]

# house_data = np.loadtxt("house/dataset.gz2")
# ier_data = np.loadtxt("ier/data.dat")

cons_prediction = IEC(dataset[:current_time]).predict([cons_algkey])
prod_prediction = IER(dataset, historical_offset).predict([prod_algkey])

charging_dictionary = {
    0: 0,
    1: 3,
    2: 14.5,
    3: 46.5
}


def calc_usage_cost(time, interval, ev_charge):
    """
    Calculates expected cost of house, ev, ier
    :param time: when we start charging
    :param interval: for how long we charge
    :param ev_charge: how much we charge the ev
    :return:
    """
    k = ev_charge

    m2 = prod_prediction[time:time + interval].sum()[prod_algkey]
    m1 = cons_prediction[time:time + interval].sum()[cons_algkey]

    s2 = prod_prediction[time:time + interval].mean()[prod_algkey_var]
    s1 = cons_prediction[time:time + interval].mean()[cons_algkey_var]

    a = (buy_price * sqrt(s1 ** 2 + s2 ** 2)) / (
        math.e ** ((k + m1 - m2) ** 2 / (2 * (s1 ** 2 + s2 ** 2))) * sqrt(2 * math.pi))

    b = (sell_price * sqrt(s1 ** 2 + s2 ** 2)) / (
        math.e ** ((k + m1 - m2) ** 2 / (2 * (s1 ** 2 + s2 ** 2))) * sqrt(2 * math.pi))

    c = ((buy_price + buy_price * erf((k + m1 - m2) / (sqrt(2) * sqrt(s1 ** 2 + s2 ** 2))) + sell_price * erfc(
        (k + m1 - m2) / (sqrt(2) * sqrt(s1 ** 2 + s2 ** 2))))) / 2
    expected = (
        a -
        b +
        (k + m1 - m2) * c
    )

    return expected


def calc_demand_cost(max_demand):
    return max_demand * 2


def calc_max_demand(time, interval, ev_charge):
    demand = cons_prediction[time:time + interval][cons_algkey] - prod_prediction[time:time + interval][prod_algkey]
    demand += ev_charge / interval

    return max(demand.max(), 0)


def calc_charge(action, interval, cur_charge, max_charge):
    # Given that Charging rates are in kW and Interval is in minutes, returns joules

    charge = min(
        charging_dictionary[action] * 1000 * interval * 60,
        max_charge - cur_charge
    )
    return charge


def min_charging_cost(charge):
    return sell_price * charge


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


def shortest_path(g, start_node, action_set, interval, target):
    """
    Creates our graph using DFS while we determine the best path
    :param g: The graph object we will add to
    :param start_node: the node we are currently on
    :param action_set: the actions possible
    :param interval: interval in minutes
    :param target: the target node
    """
    if start_node.time >= target.time:
        if start_node.battery < target.battery:
            g.add_node(start_node, usage_cost=np.inf, demand_cost=np.inf, best_action=None, max_demand=np.inf)
        return

    # shuffle(action_set)
    for action in action_set:
        charge_amount = calc_charge(action, interval, start_node.battery, target.battery)
        new_node = Node(
            battery=start_node.battery + charge_amount,
            time=start_node.time + interval
        )


        # there are many instances where we can prune this new node
        # 1. if there's no time left to charge..
        charge_max = calc_charge(max(action_set), target.time - new_node.time, new_node.battery, target.battery)
        if new_node.battery + charge_max < target.battery:
            continue  # skip
        """
        # 2. if we are guaranteed to generate a more expensive path
        ideal_remaining_cost = (calc_usage_cost(new_node.time, target.time - new_node.time, 0)
                                + g.node[start_node]['cost']
                                + interval_usage_cost
                                + min_charging_cost(target.battery - new_node.battery)
                                )

        if g.node[root]['cost'] < ideal_remaining_cost:
            continue
        """

        interval_usage_cost = calc_usage_cost(start_node.time, interval, charge_amount)
        interval_demand = calc_max_demand(start_node.time, interval, charge_amount)

        if new_node not in g:
            g.add_node(new_node, demand_cost=np.inf, usage_cost=np.inf, best_action=None, max_demand=np.inf)
            shortest_path(g, new_node, action_set, interval, target)

        this_path_usage_cost = g.node[new_node]['usage_cost'] + interval_usage_cost
        this_path_demand = max(g.node[new_node]['max_demand'], interval_demand)
        this_path_demand_cost = calc_demand_cost(this_path_demand)

        this_path_cost = this_path_usage_cost + this_path_demand_cost



        g.add_edge(start_node,
                   new_node,
                   action=action
                   )

        if this_path_cost < g.node[start_node]['usage_cost'] + g.node[start_node]['demand_cost']:
            # replace the costs of the current node
            g.add_node(start_node,
                       best_action=new_node,
                       usage_cost=this_path_usage_cost,
                       demand_cost=this_path_demand_cost,
                       max_demand=this_path_demand
                       )


def reconstruct_path(G, root):
    cur = root
    path = [root]

    while G.node[cur]['best_action'] is not None:
        cur = G.node[cur]['best_action']
        path.append(cur)

    return path


if __name__ == '__main__':
    G = nx.DiGraph()
    root = Node(0, 0)

    interval = 10
    max_depth = 25

    charging_time_perc = 0.7

    target_time = max_depth * interval

    action_set = [0, 1, 2]

    target_charge = 1000 * 60 * charging_dictionary[max(action_set)] * target_time * charging_time_perc

    print("Target Charge: {0}".format(target_charge))

    target = Node(target_charge, target_time)

    G.add_node(root, usage_cost=np.inf, demand_cost=np.inf, best_action=None, max_demand=np.inf)
    G.add_node(target, usage_cost=0, demand_cost=0, best_action=None, max_demand=0)

    print("Starting algorithm...")
    shortest_path(G, root, action_set=action_set, interval=interval, target=target)
    print("Done.")
    path = reconstruct_path(G, root)
    path_edges = list(zip(path, path[1:]))

    fig = plotly_figure(G, path=path)
    py.plot(fig)

    print(len(G.nodes()))

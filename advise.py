"""
EV advise unit
"""

import math
from math import sqrt
from random import shuffle

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

# house_data = np.loadtxt("house/dataset.gz2")
# ier_data = np.loadtxt("ier/data.dat")

cons_prediction = IEC(dataset[:(-historical_offset)]).predict([cons_algkey])

prod_prediction = IER(dataset, historical_offset).predict([prod_algkey])

charging_dictionary = {
    0: 0,
    1: 3,
    2: 14.5,
    3: 46.5
}


def calc_cost(time, interval, ev_charge):
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


def create_graph(G, start_node, action_set, interval, target):
    if start_node.time >= target.time:
        if start_node.battery >= target.battery:
            G.add_edge(start_node, target, weight=0)
            weight = G.node[start_node]['cost']

            if G.node[target]['cost'] > weight:
                G.add_node(target, cost=weight, previous=start_node)
        return

    if start_node.battery >= target.battery and min(action_set) >= 0:  # in case we charged the battery and no V2G
        action_set = [0]

    shuffle(action_set)
    for action in action_set:

        charge_amount = calc_charge(action, interval, start_node.battery, target.battery)

        new_node = Node(
            battery=start_node.battery + charge_amount,
            time=start_node.time + interval
        )

        # check if after this node we have enough time to charge the vehicle (if needed..)
        charge_max = calc_charge(max(action_set), target.time - new_node.time, new_node.battery, target.battery)
        if new_node.battery + charge_max < target.battery:
            # print("Skipping.. Not enough time to charge.")
            continue  # if so, skip this action

        edge_weight = calc_cost(start_node.time, interval, charge_amount)
        total_weight = edge_weight + G.node[start_node]['cost']

        # alpha pruning:
        alpha = (
            calc_cost(new_node.time, target.time - new_node.time, 0) +
            total_weight +
            min_charging_cost(target.battery - new_node.battery)
        )

        if alpha > G.node[target]['cost']:
            # no point in traversing this path
            # print("Skipping.. Pruned")
            continue

        if new_node not in G:
            G.add_node(new_node, cost=total_weight, previous=start_node)
        elif G.node[new_node]['cost'] > total_weight:
            G.add_node(new_node, cost=total_weight, previous=start_node)
        else:
            continue

        G.add_edge(
            start_node,
            new_node,
            weight=edge_weight,
            action=action
        )
        create_graph(G, new_node, action_set, interval, target=target)


def reconstruct_path(G, target):
    cur = target
    path = [target]

    while G.node[cur]['previous'] is not None:
        path.append(G.node[cur]['previous'])
        cur = G.node[cur]['previous']

    path.reverse()
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

    G.add_node(root, cost=0, previous=None)
    G.add_node(target, cost=np.inf, previous=None)

    create_graph(G, root, action_set=action_set, interval=interval, target=target)

    path = reconstruct_path(G, target)
    path_edges = list(zip(path, path[1:]))



    fig = plotly_figure(G, path=path)
    py.plot(fig)

    print(len(G.nodes()))

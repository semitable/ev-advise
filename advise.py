#######################################################################################
#																					  #
#								  Pre-Heating Graph								  	  #
#																					  #
#######################################################################################
###################################### imports ########################################
#import networkx as nx

'''
[TODO]
Fix maximum recursion limit.
import sys
sys.setrecursionlimit(10000) # 10000 is an example, try with different values
'''
#import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from ier.IERhybrid import renesHybrid, renes
from house.iec import IEC
from utils.utils import as_pandas
import matplotlib.pyplot as plt

####### import Graphviz


Pbuy = 1
Psell = 0.8


house_data = np.loadtxt("house/dataset.gz2")
ier_data = np.loadtxt("ier/data.dat")

cons_algkey = "Baseline Finder" #house consumption algorithm key

historical_offset = 2000

#make my predictions

prod_prediction = pd.DataFrame(renes(HistoricalOffset=historical_offset), columns = ['Time', 'Production'])
prod_prediction['Time'] = pd.to_datetime(prod_prediction['Time'], unit='s')
prod_prediction = prod_prediction.set_index('Time')


cons_prediction = as_pandas(IEC(house_data[:(-historical_offset)]).predict([cons_algkey]))

#print(cons_prediction)

#######################################################################################
#################################### Functions ########################################



#print(cons_prediction)
#print(prod_prediction)
charging_dictionary = {
    0 : 0,
    1 : 3,
    2 : 14.5,
    3 : 46.5
}

def calc_cost(time, interval, action, ev_charge):


    prod = prod_prediction[time:time+interval].sum()['Production']
    cons = cons_prediction[time:time+interval].sum()[cons_algkey] + ev_charge
    #print(prod, cons)

    return cons*Pbuy - prod*Psell



def calc_charge(action, interval, cur_charge, max_charge):
    #Given that Charging rates are in kW and Interval is in minutes, returns joules

    charge = min(
        charging_dictionary[action]*1000*interval*60,
        max_charge-cur_charge
    )
    return charge



class Node():

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

    if(start_node.time >= target.time):
        if start_node.battery >= target.battery:
            G.add_edge(start_node, target, weight = 0)
            weight = G.node[start_node]['cost']

            if G.node[target]['cost'] > weight:
                G.add_node(target, cost=weight, previous=start_node)
        return


    for action in action_set:

        charge_amount = calc_charge(action, interval, start_node.battery, target.battery)

        new_node = Node(
            battery= start_node.battery + charge_amount,
            time= start_node.time + interval
        )




        edge_weight = calc_cost(start_node.time, interval, action, charge_amount)
        total_weight = edge_weight + G.node[start_node]['cost']


        if new_node not in G:
            G.add_node(new_node, cost=total_weight, previous=start_node)
        elif G.node[new_node]['cost'] > total_weight:
            G.add_node(new_node, cost=total_weight, previous=start_node)

        G.add_edge(
            start_node,
            new_node,
            weight = edge_weight,
            action = action
        )
        create_graph(G, new_node, action_set, interval, target=target)


def reconstruct_path(G, target):
    cur = target
    path = []

    while G.node[cur]['previous'] is not None:
        print(cur)
        path.append(G.node[cur]['previous'])
        cur = G.node[cur]['previous']

    path.reverse()
    return path

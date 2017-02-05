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

import matplotlib.pyplot as plt

####### import Graphviz

try:
    import pygraphviz
    from networkx.drawing.nx_agraph import graphviz_layout
    print("using package pygraphviz")
except ImportError:
    try:
        import pydotplus
        from networkx.drawing.nx_pydot import graphviz_layout
        print("using package pydotplus")
    except ImportError:
        print()
        print("Both pygraphviz and pydotplus were not found ")
        print("see http://networkx.github.io/documentation"
              "/latest/reference/drawing.html for info")
        print()
        raise

'''
Utility functions
'''
def as_pandas(result):
    time = result[list(result.keys())[0]][:, 0]
    df = pd.DataFrame(time, columns = ['Time'])
    df['Time'] = pd.to_datetime(df['Time'], unit='s')


    for key in result:
        df[key] = result[key][:, 1].tolist()

    df = df.set_index('Time')

    return df



#######################################################################################
#################################### Functions ########################################
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

print(cons_prediction)


#print(cons_prediction)
#print(prod_prediction)

def ChargingDictionary(x):
    # Slow charging (up to 3kW), Fast charging (7-22kW), and Rapid charging units (43-50kW) (https://www.zap-map.com/charge-points/basics/)
    return {0 : 0,
            1 : 3,
            2 : 14.5,
            3 : 46.5}[x] #kW


def calc_cost(time, interval, action, battery):


    prod = prod_prediction[time:time+interval].sum()['Production']
    cons = cons_prediction[time:time+interval].sum()[cons_algkey] + ChargingDictionary(action)*1000*(interval*60)
    #print(prod, cons)



    return cons*Pbuy - prod*Psell



def charge(Action,Interval):
    #Given that Charging rates are in kW and Interval is in minutes, returns joules
    return (ChargingDictionary(Action)*1000)*(Interval*60) #Joules

def Prod(Timestep,Horizon,Interval):
    AdditionalProd=0
    for i in range(Timestep+Interval,Horizon,Interval):
        AdditionalProd+=i
    return AdditionalProd

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
        new_node = Node(
            battery= start_node.battery + charge(action, interval),
            time= start_node.time + interval
        )

        edge_weight = calc_cost(start_node.time, interval, action, start_node.battery)
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

#main Function
def DFS(G, Node, ActionSet=[0,1], Interval=1, Horizon=2, BatteryTarget=10, EV2G=False):

    timestep = G.node[Node]["timestep"]
    battery = G.node[Node]["battery"]

    print("Current State: ", timestep, battery)

    for Action in ActionSet:

        next_battery = battery + charge(Action, Interval) + charge(max(ActionSet),(Horizon-timestep-Interval))

        print("---- Child(", Action, "): ", timestep, next_battery)

        if next_battery >= BatteryTarget:

            interval_cost = Cost(timestep, Interval, Action, battery, BatteryTarget)

            #print (interval_cost-Prod(timestep,Horizon,Interval),G.node[Node]['MininumAdditionalCost'])

            if EV2G or (not EV2G and interval_cost-Prod(timestep,Horizon,Interval)<=G.node[Node]['MininumAdditionalCost']):
                BatteryNew = battery + charge(Action,Interval)
                TimestepNew = timestep+Interval

                NodeNew = str(BatteryNew)+"-"+str(TimestepNew)

                if not G.has_node(NodeNew):
                    if TimestepNew<Horizon and BatteryNew<BatteryTarget:
                        G.add_node(NodeNew, battery=BatteryNew, timestep=TimestepNew, MininumAdditionalCost=float("Inf"), BestAction=float("Inf"))
                        G.add_edge(Node, NodeNew, weight=interval_cost)
                        G = DFS(G, NodeNew, ActionSet, Interval, Horizon, BatteryTarget)
                    elif BatteryNew>=BatteryTarget:
                        G.add_node(NodeNew, battery=BatteryNew, timestep=TimestepNew, MininumAdditionalCost=0, BestAction=float("Inf"))
                        G.add_edge(Node, NodeNew, weight=interval_cost)
                else:
                    G.add_edge(Node, NodeNew, weight=interval_cost)

                # Propagate Mininum Additional Cost for each Node and assign Best Action
                if G.node[NodeNew]['MininumAdditionalCost'] + interval_cost <= G.node[Node]['MininumAdditionalCost']:
                    G.node[Node]['MininumAdditionalCost'] = G.node[NodeNew]['MininumAdditionalCost'] + interval_cost
                    G.node[Node]['BestAction'] = Action
        else:
            print(next_battery, BatteryTarget)

    return G






########## Callable Controller
G=nx.DiGraph()
root = Node(0, 0)


max_depth = 5
interval = 1
target_charge = 1000000

target = Node(target_charge, max_depth)

G.add_node(root, cost=0, previous=None)
G.add_node(target, cost=np.inf, previous=None)

create_graph(G, root, action_set=[0, 1, 2], interval=1, target=target)

print(G.node[target])

path = reconstruct_path(G, target)
path_edges = list(zip(path, path[1:]))

# Call DFS Graph


#print DFS(G, Root, ActionSet=[0,1], Interval=1, Horizon=10, BatteryTarget=4).node[Root]['BestAction']

plt.title("draw_networkx")
pos=graphviz_layout(G,prog='dot')
nx.draw(G,pos,with_labels=True,arrows=False, node_color='k')
#labels = nx.get_edge_attributes(G,'weight')
#nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)

nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='r')
nx.draw_networkx_edges(G,pos,edgelist=path_edges,edge_color='r')

labels = nx.get_node_attributes(G,'BestAction')
for k,v in labels.items():
    plt.text(pos[k][0],pos[k][1]+20,s="Best Action: "+str(labels[k]), bbox=dict(facecolor='red', alpha=0.5),horizontalalignment='center')

plt.show()

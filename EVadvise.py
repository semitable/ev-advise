#######################################################################################
#																					  #
#								  Pre-Heating Graph								  	  #
#																					  #
#######################################################################################
###################################### imports ########################################
import networkx as nx

'''
[TODO]
Fix maximum recursion limit.
import sys
sys.setrecursionlimit(10000) # 10000 is an example, try with different values
'''

import numpy as np

import plotly.graph_objs as go
try:
    import pygraphviz
    from networkx.drawing.nx_agraph import graphviz_layout
    #print("using package pygraphviz")
except ImportError:
    try:
        import pydotplus
        from networkx.drawing.nx_pydot import graphviz_layout
        #print("using package pydotplus")
    except ImportError:
        print()
        print("Both pygraphviz and pydotplus were not found ")
        print("see http://networkx.github.io/documentation"
              "/latest/reference/drawing.html for info")
        print()
        raise


def plotly_figure(G, path=None):
    pos = graphviz_layout(G, prog='dot')

    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=go.Line(width=0.5, color='#888'),
        hoverinfo=['none'],
        mode='lines')

    my_annotations = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += [x0, x1, None]
        edge_trace['y'] += [y0, y1, None]

        my_annotations.append(
            dict(
                x=(x0+x1)/2,
                y=(y0+y1)/2,
                xref='x',
                yref='y',
                text='{0:.2f}'.format(G.get_edge_data(edge[0], edge[1])['weight']),
                showarrow=False,
                arrowhead=2,
                ax=0,
                ay=0
            )
        )



    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=go.Marker(
            showscale=False,
            # colorscale options
            # 'Greys' | 'Greens' | 'Bluered' | 'Hot' | 'Picnic' | 'Portland' |
            # Jet' | 'RdBu' | 'Blackbody' | 'Earth' | 'Electric' | 'YIOrRd' | 'YIGnBu'
            colorscale='YIGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line=dict(width=2)))

    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'].append(x)
        node_trace['y'].append(y)

        node_info = "Time: +{0}<br>Battery: {1}<br>Minimum Additional Cost: {2}<br>Best Action: {3}".format(G.node[node]['Timestep'], G.node[node]['Battery'], G.node[node]['MininumAdditionalCost'], G.node[node]['BestAction'])

        node_trace['text'].append(node_info)

        if path is None:
            node_trace['marker']['color'].append(G.node[node]['MininumAdditionalCost'])
        elif node in path:
            node_trace['marker']['color'].append('rgba(255, 0, 0, 1)')
        else:
            node_trace['marker']['color'].append('rgba(0, 0, 255, 1)')


    fig = go.Figure(data=go.Data([edge_trace, node_trace]),
                    layout=go.Layout(
                        title='<br>Network graph made with Python',
                        titlefont=dict(size=16),
                        showlegend=False,
                        width=650,
                        height=650,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=my_annotations,
                        xaxis=go.XAxis(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=go.YAxis(showgrid=False, zeroline=False, showticklabels=False)))
    return fig



from ier.ier import  IER
from house.iec import IEC
from utils.utils import as_pandas

####### import Graphviz


Pbuy =  1.2
Psell = 0.8



house_data = np.loadtxt("house/dataset.gz2")
ier_data = np.loadtxt("ier/data.dat")

cons_algkey = "Baseline Finder Hybrid" #house consumption algorithm key
prod_algkey = "Renes Hybrid"

historical_offset = 2000

#make my predictions

prod_prediction = as_pandas(IER(ier_data, historical_offset).predict([prod_algkey]))
cons_prediction = as_pandas(IEC(house_data).predict([cons_algkey]))

#######################################################################################
#################################### Functions ########################################

charging_dictionary = {
    0 : 0,
    1 : 3,
    2 : 14.5,
    3 : 46.5
}


def Cost(Timestep,Interval, ev_charge):

    if Interval == 0:
        return 0

    prod = prod_prediction[Timestep:Timestep + Interval].sum()[prod_algkey]
    cons = cons_prediction[Timestep:Timestep + Interval].sum()[cons_algkey] + ev_charge

    return cons * Pbuy - prod * Psell


def Charge(Action,Interval, cur_charge, max_charge):
    charge = min(
        charging_dictionary[Action]*1000*Interval*60,
        max_charge-cur_charge
    )
    return charge

def Prod(Timestep,Horizon,Interval):
    AdditionalProd=0
    for i in range(Timestep+Interval,Horizon,Interval):
        AdditionalProd+=i
    return AdditionalProd


#main Function
def DFS(G, Node, ActionSet=[0,1], Interval=1, Horizon=2, BatteryTarget=10, EV2G=False):

    Timestep = G.node[Node]["Timestep"]
    Battery = G.node[Node]["Battery"]

    for Action in ActionSet:
        this_charge = Charge(Action, Interval, Battery, BatteryTarget)
        max_remaining_charge = (Charge(max(ActionSet), (Horizon-Timestep-Interval), this_charge+Battery, BatteryTarget))

        if Battery + this_charge + max_remaining_charge >= BatteryTarget:
            CostInterval = Cost(Timestep,Interval, this_charge)
            #print CostInterval-Prod(Timestep,Horizon,Interval),G.node[Node]['MininumAdditionalCost']
            if EV2G or (not EV2G and CostInterval-Prod(Timestep,Horizon,Interval)<=G.node[Node]['MininumAdditionalCost']):
                BatteryNew = Battery + this_charge
                TimestepNew = Timestep+Interval

                NodeNew = str(BatteryNew)+"-"+str(TimestepNew)

                if not G.has_node(NodeNew):
                    if TimestepNew<Horizon and BatteryNew<BatteryTarget:
                        G.add_node(NodeNew, Battery=BatteryNew, Timestep=TimestepNew, MininumAdditionalCost=float("Inf"), BestAction=float("Inf"))
                        G.add_edge(Node, NodeNew, weight=CostInterval)
                        G = DFS(G, NodeNew, ActionSet, Interval, Horizon, BatteryTarget)
                    elif BatteryNew>=BatteryTarget:
                        G.add_node(NodeNew, Battery=BatteryNew, Timestep=TimestepNew, MininumAdditionalCost=Cost(TimestepNew, (Horizon-TimestepNew)*Interval, 0), BestAction=float("Inf"))
                        G.add_edge(Node, NodeNew, weight=CostInterval)
                else:
                    G.add_edge(Node, NodeNew, weight=CostInterval)

                # Propagate Mininum Additional Cost for each Node and assign Best Action
                if G.node[NodeNew]['MininumAdditionalCost'] + CostInterval <= G.node[Node]['MininumAdditionalCost']:
                    G.node[Node]['MininumAdditionalCost'] = G.node[NodeNew]['MininumAdditionalCost'] + CostInterval
                    G.node[Node]['BestAction'] = Action

    return G


########## Callable Controller

Battery=0
Timestep=0
G=nx.DiGraph()
Root = str(Battery)+"-"+str(Timestep)
G.add_node(Root, Battery=Battery, Timestep=Timestep, MininumAdditionalCost=float("Inf"), BestAction=float("Inf"))

# Call DFS Graph
G = DFS(G, Root, ActionSet=[0,1,2], Interval=1, Horizon=15, BatteryTarget=1000000)

print(len(G.nodes()))

#print DFS(G, Root, ActionSet=[0,1], Interval=1, Horizon=10, BatteryTarget=4).node[Root]['BestAction']
import plotly.offline as py

fig = plotly_figure(G)
py.plot(fig)
import pandas as pd
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

'''
Utility functions
'''
def as_pandas(result):
    time = result[list(result.keys())[0]][:, 0]
    df = pd.DataFrame(time, columns = ['Time'])
    df['Time'] = pd.to_datetime(df['Time'], unit='s')


    for key in result:
        df[key] = result[key][:, 1].tolist()
        try:
            df[key + " Var"] = result[key][:, 2].tolist()
        except IndexError:
            pass

    df = df.set_index('Time')
    return df



def plotly_figure(G, path=None):
    pos = graphviz_layout(G, prog='dot')

    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=go.Line(width=0.5, color='#888'),
        hoverinfo='none',
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
                text='{0:.2f}'.format(G.get_edge_data(edge[0], edge[1])['action']),
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

        node_info = "Time: +{0}<br>Battery: {1}<br>Usage Cost: {2}<br>Maximum Demand: {3}".format(node.time,
                                                                                                  node.battery,
                                                                                                  G.node[node][
                                                                                                      'usage_cost'],
                                                                                                  G.node[node][
                                                                                                      'max_demand'])

        node_trace['text'].append(node_info)

        if path is None:
            node_trace['marker']['color'].append(G.node[node]['usage_cost'])
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

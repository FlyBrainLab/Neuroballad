import networkx as nx

def visualize_video(C, name, config={}, visualization_variable='V',
                    out_name='test.avi'):
    """
    Visualizes all ID's using a set visualization variable over time,
    saving them to a video file.

    Example
    --------
    >>> C.visualize_video([0, 2, 4], out_name='visualization.avi')
    """
    uids = []
    if config == {}:
        config = {'variable': visualization_variable, 'type': 'waveform',
                  'uids': [C.Viz._uids['lpu'][visualization_variable]]}
    for i in name:
        uids.append(i)
    config['uids'] = uids
    C.Viz.codec = 'mpeg4'
    C.Viz.add_plot(config, 'lpu')
    C.Viz.update_interval = 1e-4
    C.Viz.out_filename = out_name
    C.Viz.run()

def visualize_circuit(C, prog='dot', splines='line', view=False,
                      filename='neuroballad_temp_circuit.svg', format='svg'): 
    '''Visualize Circuit as Graph
    
    Parameters
    ----------
    C: neuroballad.Circuit
        Circuit to be plotted
    prog: str
        GraphViz layouting algorithm, default to 'dot'
    splines: str, ['none', 'polyline', 'line', 'ortho', 'splines', 'curved']
        Lines to be used in GraphViz, default to 'line'
    filename: str
    '''
    styles = {
        'graph': {
            'label': C.name,
            # 'fontname': 'LM Roman 10',
            'fontsize': '16',
            'fontcolor': 'black',
            # 'bgcolor': '#333333',
            # 'rankdir': 'LR',
            'splines': splines,
            'model': 'circuit',
            'size': '250,250',
            'overlap': 'false',
        },
        'nodes': {
            # 'fontname': 'LM Roman 10',
            'shape': 'box',
            'fontcolor': 'black',
            'color': 'black',
            'style': 'rounded',
            'fillcolor': '#006699',
            # 'nodesep': '1.5',
        },
        'edges': {
            'style': 'solid',
            'color': 'black',
            'arrowhead': 'open',
            'arrowsize': '0.5',
            'fontname': 'Courier',
            'fontsize': '12',
            'fontcolor': 'black',
            'splines': 'ortho',
            'concentrate': 'false',
        }
    }
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    G = C.G
    classnames = list(
        set([item for key, item in nx.get_node_attributes(G, 'class').items()]))
    colors = plt.cm.get_cmap('Spectral', len(classnames))
    # G.remove_nodes_from(nx.isolates(G))
    mapping = {}
    node_types = set()
    for n, d in G.nodes(data=True):
        node_types.add(d['name'].rstrip('1234567890'))
    node_nos = dict.fromkeys(node_types, 1)
    for n, d in G.nodes(data=True):
        node_type = d['name'].rstrip('1234567890')
        mapping[n] = d['name'].rstrip(
            '1234567890') + str(node_nos[node_type])
        node_nos[node_type] += 1
    G = nx.relabel_nodes(G, mapping)
    A = nx.drawing.nx_agraph.to_agraph(G)
    #A.graph_attr['fontname']= 'LM Roman 10'
    #A.graph_attr['splines'] = 'ortho'
    # A.graph_attr['bgcolor'] = '#333333'
    A.graph_attr.update(styles['graph'])
    A.write('file.dot')
    for i in A.edges():
        e = A.get_edge(i[0], i[1])
        #e.attr['splines'] = 'ortho'
        e.attr.update(styles['edges'])
        e.attr.update(label=e.attr['variable'])
        if i[0][:-1] == 'Repressor':
            e.attr['arrowhead'] = 'tee'
    for i in A.nodes():
        n = A.get_node(i)
        #n.attr['shape'] = 'box'
        n.attr.update(styles['nodes'])
        c = colors(classnames.index(n.attr['class']))
        n.attr.update(style='rounded,filled',
                      fillcolor=matplotlib.colors.rgb2hex(c))
    A.layout(prog=prog)
    if filename is not None:
        A.draw(filename, format=format)
    if view:
        return A.draw(None, format=format)
    return A

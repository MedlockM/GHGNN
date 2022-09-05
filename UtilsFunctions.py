from collections import defaultdict
import networkx as nx
#from networkx.algorithms import community
import community as community_louvain
import numpy as np

def build_hyp_as_communities(adj_lists):
    hyp2nodes = {}
    G = nx.Graph(adj_lists)
    #print(list(G.edges))
    comm = community_louvain.best_partition(G)
    #print(comm)
    #print(len(comm))
    #print(max(comm.keys()))
    #print(max(comm.values()))
    for i in range(max(comm.values())+1):
        hyp2nodes[i]=[k for k in comm.keys() if comm[k] == i]
    #hyp2nodes = {k: v for (k,v) in zip(range(len(comm)), comm)}
    #print(hyp2nodes)
    return hyp2nodes
def build_hyp_as_features(feat_data):
    '''
    Compute hyperedges sets from one hot encoded categorical nodes features matrix (each hyperedge represent a
    feature, each node having this feature will be in this hyperedge)
    :param feat_data: one hot encoded categorical nodes features matrix
    :return: hyperedges2nodes: dictionnary mapping each hyperedge to his internal nodes
    '''
    hyperedge2nodes = {}
    for i in range(len(feat_data[0])):
        mask = np.transpose(feat_data)[i]
        hyperedge2nodes[i] = [j for j in range(len(feat_data)) if mask[j]==1]
    return hyperedge2nodes

def build_hyp_as_neighbors(adj_lists):
    '''
    Compute hyperedges sets from adjacency matrix (for each node, his 1-hop neighbordhood is considered as a hyperedge)
    :param adj_lists: adjacency matrix represented as a dictionnary (key: node name, value: list of neighbors name)
    :return: hyperedges2nodes: dictionnary mapping each hyperedge to his internal nodes
    '''

    max_groupsize = 169 # fix a upper limit of hyperedge size
    min_groupsize = 2 # fix a sub limit of hyperedge size
    hyperedge2nodes = defaultdict(set)
    max_size = 0
    min_size = 100000
    for i, (node, neighborhood) in enumerate(adj_lists.items()):
        if len(neighborhood) + 1 <= max_groupsize and len(neighborhood) + 1 >= min_groupsize:
            hyperedge2nodes[i] = list(neighborhood)
        if len(neighborhood) + 1 < min_size:
            min_size = len(neighborhood) + 1
        if len(neighborhood) > max_size:
            max_size = len(neighborhood) + 1
    #print(min_size, max_size)
    #print(len(list(hyperedge2nodes.keys())))
    #print(hyperedge2nodes)
    return hyperedge2nodes
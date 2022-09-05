import os
import time
import random
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

from HyperedgeAggregator import *
from NodeEncoder import NodeEncoder
from NodeAggregator import *
from HyperedgeEncoder import HyperedgeEncoder
from Model import SpatialHCONV
from UtilsFunctions import *
import matplotlib.pyplot as plt

def load_cora(dataset, nodes_data_path, edges_data_path):
    '''
    function to load Cora dataset from nodes dataset and edges data

    :parameter:
            nodes_data_path (str): local path to nodes data file ('cora.content')
            edges_data_path (str): local path to edges data (nodes pairing) file ('cora.cites')
    :returns:
            feat_data (numpy array) : array of shape (num_nodes, num_feats) representing features by node
            labels (int numpy array): array of shape (num_nodes, 1) representing label by node
            adj_lists (dictionnary): key: node id, value: set of neighbors ids
    '''

    if dataset == 'cora':
        num_nodes = 2708
        num_feats = 1433
    elif dataset == 'citeseer':
        num_nodes = 3312
        num_feats = 3703
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes,1), dtype=np.int64)
    node_map = {}
    label_map = {}
    
    with open(nodes_data_path) as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            #print("info", info[1:-1])
            feat_data[i,:] = [float(s) for s in info[1:-1]]
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]
    #print(node_map)
    adj_lists = defaultdict(set)
    with open(edges_data_path) as fp:
        for i,line in enumerate(fp):
            #print(i, line)
            info = line.strip().split()
            #print(info)
            try:
                paper1 = node_map[info[0]]
                paper2 = node_map[info[1]]
                adj_lists[paper1].add(paper2)
                adj_lists[paper2].add(paper1)
            except KeyError:
                pass
    return feat_data, labels, adj_lists

def run_hyper_cora(dataset, h1_dim, h2_dim=1, build_hyperedges_criteria='as_neighbors', hyperedge_aggregator1_type='mean',
                   node_aggregator1_type='mean', hyperedge_aggregator2_type=None, node_aggregator2_type=None):
    '''
    load Cora dataset, build hyperedges according to user criteria, create 2 layers of HyperGraph Neural Network
    with Hyperedges Embedding and run 100 batch iterations on nodes classification task on Cora hypergraph,
    display averaged F1 score on validation subset and batch training time.

    :parameter:
            build_hyperedges_criteria (str): how to build hyperedges from nodes, choose between:
                'as_neighbors'=each neighbors set is considered as an hyperedge
                'as_features'=each hyperedge represent a categorical feature (if a node has this attribute, it is in the
                              corresponding hyperedge)
            hyperedge_aggregator1_type (str): hyperedege aggregator type of first convolutionnal layer, choose between:
                'mean'=mean aggregator
                'max'=max pooling aggregator
            node_aggregator1_type (str): node aggregator type of first convolutionnal layer, choose between:
                'mean'=mean aggregator
                'max'=max pooling aggregator
            hyperedge_aggregator2_type (str): hyperedege aggregator type of second convolutionnal layer, choose between:
                'mean'=mean aggregator
                'max'=max pooling aggregator
                None=no second convolutionnal layer
            node_aggregator2_type (str): node aggregator type of second convolutionnal layer, choose between:
                'mean'=mean aggregator
                'max'=max pooling aggregator
                None=no second convolutionnal layer
    :return:
    '''

    random.seed(1)
    if dataset == 'cora':
        num_nodes = 2708
        num_feats = 1433
        num_classes = 7
    elif dataset == 'citeseer':
        num_nodes = 3312
        num_feats = 3703
        num_classes = 6
    root_dir = os.path.dirname(os.path.abspath(__file__))
    nodes_data_path = root_dir+"/"+dataset+".content"
    edges_data_path = root_dir+"/"+dataset+".cites"
    feat_data, labels, adj_lists = load_cora(dataset, nodes_data_path, edges_data_path)
    #print("adj lists", adj_lists)
    #print("features data", feat_data)
    #print("num samples =", len(feat_data), "dim =", len(feat_data[0]))
    features = nn.Embedding(num_nodes, num_feats)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)

    if build_hyperedges_criteria=='as_neighbors':
        hyperedge2nodes = build_hyp_as_neighbors(adj_lists)
    if build_hyperedges_criteria=='as_features':
        hyperedge2nodes = build_hyp_as_features(feat_data)
    if build_hyperedges_criteria=='as_communities':
        hyperedge2nodes = build_hyp_as_communities(adj_lists)

    node2hyperedges = {}
    for j in range(len(feat_data)):
        node2hyperedges[j] = [i for i in hyperedge2nodes.keys() if j in hyperedge2nodes[i]]

    hypere_features = nn.Embedding(len(hyperedge2nodes), num_feats)
    hypere_features.weight = nn.Parameter(torch.mul(torch.ones(len(hyperedge2nodes), num_feats, requires_grad=False), 1/num_feats))
    #print(hypere_features.weight)
    #nn.init.xavier_uniform_(hypere_features.weight.data)
    #print(hypere_features.weight)

    if hyperedge_aggregator1_type == 'mean':
        hyperedge_agg = HyperedgeMeanAggregator(features, hyperedge2nodes)
    if hyperedge_aggregator1_type == 'max':
        hyperedge_agg = HyperedgeMaxAggregator(features, hyperedge2nodes)
    if hyperedge_aggregator1_type == 'att':
        hyperedge_enc = HyperedgeAttAggregator(features, hyperedge2nodes, num_feats, h1_dim, hypere_features)
        #hyperedge_enc = HyperedgeEncoder(128, 128, hyperedge_agg, lambda hyperedge: hyperedge_agg.embed_h(hyperedge).t())
    else:
        hyperedge_enc = HyperedgeEncoder(num_feats, h1_dim, hyperedge_agg, hypere_features)
    if node_aggregator1_type=='mean':
        node_agg = NodeMeanAggregator(hypere_features, node2hyperedges)
    if node_aggregator1_type=='max':
        node_agg = NodeMaxAggregator(hypere_features, node2hyperedges)
    if node_aggregator1_type=='att':
        node_enc = HyperedgeAttAggregator(hypere_features, node2hyperedges, num_feats, h1_dim, features)
    else:
        if hyperedge_aggregator1_type == 'att':
            node_enc = NodeEncoder(num_feats, h1_dim, node_agg, features)
        else:
            node_enc = NodeEncoder(num_feats, h1_dim, node_agg, features)

    if hyperedge_aggregator2_type != None and node_aggregator2_type != None:
        if hyperedge_aggregator2_type=='mean':
            hyperedge_agg2 = HyperedgeMeanAggregator(lambda node: node_enc.compute(node), hyperedge2nodes)
        if hyperedge_aggregator2_type=='max':
            hyperedge_agg2 = HyperedgeMaxAggregator(lambda node: node_enc.compute(node), hyperedge2nodes)
        if hyperedge_aggregator2_type=='att':
            hyperedge_enc2= HyperedgeAttAggregator(lambda node: node_enc.compute(node).t(), hyperedge2nodes, h1_dim, h2_dim,
                                                   lambda hyperedge: hyperedge_enc.compute(hyperedge))
        else:
            hyperedge_enc2 = HyperedgeEncoder(h1_dim, h2_dim, hyperedge_agg2, lambda hyperedge: hyperedge_agg2.compute(hyperedge))

        if node_aggregator2_type=='mean':
            node_agg2 = NodeMeanAggregator(lambda hyperedge: hyperedge_enc.compute(hyperedge), node2hyperedges)
        if node_aggregator2_type=='max':
            node_agg2 = NodeMaxAggregator(lambda hyperedge: hyperedge_enc.compute(hyperedge), node2hyperedges)
        if node_aggregator2_type=='att':
            node_enc2 = HyperedgeAttAggregator(lambda hyperedge: hyperedge_enc.compute(hyperedge), node2hyperedges, h1_dim,
                                              h2_dim, lambda node: node_enc.compute(node), base_model=node_enc)

        else:
            node_enc2 = NodeEncoder(h1_dim, h2_dim, node_agg2, lambda node: node_enc.compute(node), base_model=node_enc)

        graphsage = SpatialHCONV(num_classes, node_enc2)

    else:
        graphsage = SpatialHCONV(num_classes, node_enc)
    #print("model parameters", (graphsage.parameters()))
    rand_indices = np.random.permutation(num_nodes)
    ratio_test = 0.2
    test = rand_indices[:int(ratio_test*num_nodes)]
    val = rand_indices[1000:1500]
    train = list(rand_indices[int(ratio_test*num_nodes)+1:])

    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.7)
    times = []
    train_loss = []
    val_loss = []
    train_scores = []
    val_scores = []
    for batch in range(70):
        batch_nodes = train[:250]
        #print('batch nodes', batch_nodes)
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes, Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
        loss.backward()
        optimizer.step()
        #print("model params", graphsage.para)
        end_time = time.time()
        times.append(end_time-start_time)
        train_loss.append(loss)
        val_loss_value = graphsage.loss(val, Variable(torch.LongTensor(labels[np.array(val)])))
        val_loss.append(val_loss_value)
        #train_output = graphsage.compute(batch_nodes)
        #train_score = f1_score(labels[batch_nodes], train_output.data.numpy().argmax(axis=1), average="micro")
        #train_scores.append(train_score)
        #val_output = graphsage.compute(val)
        #val_score = f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro")
        #val_scores.append(val_score)
        #print(f"batch {batch}: loss = {loss.item()}")

    # plt.plot([i for i in range(300)], train_loss, label='Loss on train set')
    # plt.plot([i for i in range(300)], val_loss, label='Loss on validation set')
    # plt.xlabel('epochs')
    # plt.legend()
    # plt.show()
    # val_output = graphsage.compute(val)
    # val_score = f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro")
    test_output = graphsage.compute(test)
    test_score = f1_score(labels[test], test_output.data.numpy().argmax(axis=1), average="micro")
    print("Test F1:", f1_score(labels[test], test_output.data.numpy().argmax(axis=1), average="micro"))
    #print("Average batch time:", np.mean(times))
    batch_time = np.mean(times)
    return test_score, batch_time

if __name__ == '__main__':
    # list_agg = ['mean', 'max', 'att']
    # df_cora = pd.DataFrame(index=list_agg, columns=list_agg)
    # df_citeseer = pd.DataFrame(index=list_agg, columns=list_agg)
    # scores = []
    # for data in ['cora', 'citeseer']:
    #     print(data)
    #     for aggregator_node in list_agg:
    #         print('node agg :', aggregator_node)
    #         for aggregator_hyp in list_agg:
    #             print('hyp agg :', aggregator_hyp)
    #             for i in range(20):
    #                 scores.append(run_hyper_cora(data, build_hyperedges_criteria='as_neighbors', h1_dim=128, h2_dim=128, hyperedge_aggregator1_type=aggregator_hyp, node_aggregator1_type=aggregator_node, hyperedge_aggregator2_type=None, node_aggregator2_type=None))
    #             if data == 'cora':
    #                 df_cora[aggregator_node][aggregator_hyp]=str(np.mean(scores))+'+/-'+str(np.std(scores))
    #             if data == 'citeseer':
    #                 df_citeseer[aggregator_node][aggregator_hyp]=str(np.mean(scores))+'+/-'+str(np.std(scores))
    #
    # df_cora.to_csv(r'/Users/SB6280/Desktop/Engie_Project/Mémoire/Expériences/agg_cora.csv', sep=';')
    # df_citeseer.to_csv(r'/Users/SB6280/Desktop/Engie_Project/Mémoire/Expériences/agg_citeseer.csv', sep=';')
    # list_build = ['as_neighbors', 'as_features', 'as_communities']
    # df_cora = pd.DataFrame(columns=list_build)
    # df_citeseer = pd.DataFrame(columns=list_build)
    # scores = []
    # for data in ['cora', 'citeseer']:
    #     for build_criteria in list_build:
    #
    #         for i in range(20):
    #             scores.append(run_hyper_cora(data, build_hyperedges_criteria=build_criteria, h1_dim=128, h2_dim=128, hyperedge_aggregator1_type='mean', node_aggregator1_type='mean', hyperedge_aggregator2_type='mean', node_aggregator2_type='mean'))
    #         if data == 'cora':
    #             df_cora[build_criteria]=[str(np.mean(scores))+'+/-'+str(np.std(scores))]
    #         if data == 'citeseer':
    #             df_citeseer[build_criteria]=[str(np.mean(scores))+'+/-'+str(np.std(scores))]
    #
    # df_cora.to_csv(r'/Users/SB6280/Desktop/Engie_Project/Mémoire/Expériences/build_cora.csv', sep=';')
    # df_citeseer.to_csv(r'/Users/SB6280/Desktop/Engie_Project/Mémoire/Expériences/build_citeseer.csv', sep=';')
    # plt.close()
    # list_h = [1, 2, 5, 10, 20, 50, 100, 200, 501, 1000, 2000, 5000]
    # #list_h = [1, 20]
    # list_h1 = np.repeat(list_h, len(list_h))
    # list_h1 = list_h1.tolist()
    # list_h2 = len(list_h) * list_h
    # mean_score = []
    # times = []
    # for h1, h2 in zip(list_h1, list_h2):
    #     scores = []
    #     for i in range(20):
    #         s, t = run_hyper_cora('citeseer', build_hyperedges_criteria='as_neighbors', h1_dim=h1, h2_dim=h2,
    #                             hyperedge_aggregator1_type='mean', node_aggregator1_type='mean',
    #                            hyperedge_aggregator2_type='mean', node_aggregator2_type='mean')
    #         scores.append(s)
    #     times.append(t)
    #     mean_score.append(np.mean(scores))
    #     #print(mean_score)
    # fig_score, ax_score = plt.subplots()
    # fig_time, ax_time = plt.subplots()
    # scatter_score = ax_score.scatter(list_h2, list_h1, c=mean_score)
    # scatter_time = ax_time.scatter(list_h2, list_h1, c=times)
    # for ax in [ax_score, ax_time]:
    #     ax.set_xlabel('second layer embedding dimension')
    #     ax.set_ylabel('first layer embedding dimension')
    #     ax.set_xscale('log')
    #     ax.set_yscale('log')
    # clb_score = plt.colorbar(scatter_score, ax=ax_score)
    # clb_score.set_label('F1-Score on validation set')
    # clb_time = plt.colorbar(scatter_time, ax=ax_time)
    # clb_time.set_label('Batch training time (s)')
    # plt.show()
    # scores = []
    # for i in range(1):
    #     s, t = run_hyper_cora('citeseer', build_hyperedges_criteria='as_neighbors', h1_dim=501, h2_dim=501,
    #                         hyperedge_aggregator1_type='mean', node_aggregator1_type='mean',
    #                        hyperedge_aggregator2_type='mean', node_aggregator2_type='mean')
    #     scores.append(s)
    # mean_score = np.mean(scores)
    # std = np.std(scores)
    # print('citeseer optimized score :', mean_score)
    # print('citeseer std', std)

    # list_h = [1, 2, 5, 10, 20, 50, 100, 200, 501, 1000, 2000, 5000]
    # plt.close()
    # #list_h = [1, 10, 100]
    # times = []
    # mean_score = []
    # std = []
    # for h in list_h:
    #     scores = []
    #     for i in range(20):
    #         s, t = run_hyper_cora('cora', build_hyperedges_criteria='as_neighbors', h1_dim=h, h2_dim=1,
    #                             hyperedge_aggregator1_type='mean', node_aggregator1_type='mean',
    #                            hyperedge_aggregator2_type=None, node_aggregator2_type='mean')
    #         scores.append(s)
    #     times.append(t)
    #     mean_score.append(np.mean(scores))
    #     std.append(np.std(scores))
    #     #print(mean_score)
    # fig_score, ax_score = plt.subplots()
    # fig_time, ax_time = plt.subplots()
    # scatter_score = ax_score.errorbar(list_h, mean_score, std, capsize=3.0)
    # scatter_time = ax_time.plot(list_h, times)
    # for ax in [ax_score, ax_time]:
    #     ax.set_xlabel('layer embedding dimension')
    # ax_score.set_xscale('log')
    # ax_score.set_ylabel('F1-score on validation set with deviation')
    # ax_time.set_ylabel('Batch training time (s)')
    # plt.show()

    # s, t = run_hyper_cora('cora', build_hyperedges_criteria='as_neighbors', h1_dim=5000, h2_dim=128,
    #                             hyperedge_aggregator1_type='mean', node_aggregator1_type='mean',
    #                            hyperedge_aggregator2_type='mean', node_aggregator2_type='mean')
    # s, t = run_hyper_cora('citeseer', build_hyperedges_criteria='as_neighbors', h1_dim=501, h2_dim=128,
    #                             hyperedge_aggregator1_type='mean', node_aggregator1_type='mean',
    #                            hyperedge_aggregator2_type='mean', node_aggregator2_type='mean')

    s, t = run_hyper_cora('cora', build_hyperedges_criteria='as_neighbors', h1_dim=5000, h2_dim=5000,
                                hyperedge_aggregator1_type='att', node_aggregator1_type='att',
                               hyperedge_aggregator2_type='att', node_aggregator2_type='att')



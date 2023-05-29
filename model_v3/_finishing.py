import numpy as np
from tqdm.notebook import tqdm
from collections import defaultdict


# EDGE BASED MODEL

def get_node_delay_aggregation(self, node_list, delay_ts_matrix):
    # GET THE NODEWISE AGGREGATED DELAYS IN SIMULATION
    steps = delay_ts_matrix.shape[0]
    delay_matrix_node_aggregated = np.zeros((steps , len(node_list)))
    num_timesteps = len(delay_ts_matrix)
    for i in tqdm(range(num_timesteps), total = num_timesteps, disable=not self.is_global_verbose):
        for j in range(len(delay_ts_matrix[i])): 
            e = self.real_edge_list[j]
            index_v = node_list.index(e[1])
            delay_matrix_node_aggregated[i][index_v] += delay_ts_matrix[i][j]
    return delay_matrix_node_aggregated


def get_node_cluster_delay_aggregation(self, cluster_list, delay_ts_matrix):
    # GET THE NODEWISE AGGREGATED DELAYS IN SIMULATION
    steps = delay_ts_matrix.shape[0]
    delay_matrix_node_aggregated = np.zeros((steps, len(cluster_list)))
    num_timesteps = len(delay_ts_matrix)
    for i in tqdm(range(num_timesteps), total = num_timesteps, disable=not self.is_global_verbose):
        for j in range(len(delay_ts_matrix[i])):
            e = self.cluster_edge_list[j]
            index_v = cluster_list.index(e[1])
            delay_matrix_node_aggregated[i][index_v] += delay_ts_matrix[i][j]
    return delay_matrix_node_aggregated


def redistribute_ts_from_clusters_to_nodes(self, sim_delay_ts):
    if self.g is None:
        raise("No graph parameters were learned. Please use .calculate_model_parameters(g, cluster_dict) first! ")

    nodes_in_clusters_dict = defaultdict(list)
    for node, cluster in self.cluster_dict.items(): 
        if node not in self.node_list:
            continue
        nodes_in_clusters_dict[cluster].append(node)
    
    sim_node_delay_ts = np.zeros((sim_delay_ts.shape[0], len(self.node_list)))
    for cluster_label in self.cluster_list:
        cluster_nodes = nodes_in_clusters_dict[cluster_label]
        index_cluster = self.cluster_list.index(cluster_label)
        cluster_delay_ts = sim_delay_ts[:, index_cluster]
        for v in cluster_nodes:
            index_v = self.node_list.index(v)
            sim_node_delay_ts[:, index_v] = cluster_delay_ts/len(cluster_nodes)
    return sim_node_delay_ts

def redistribute_real_ts_from_clusters_to_nodes(self, real_cluster_timeseries):
    if self.g is None:
        raise("No graph parameters were learned. Please use .calculate_model_parameters(g, cluster_dict) first! ")

    nodes_in_clusters_dict = defaultdict(list)
    for node, cluster in self.cluster_dict.items(): 
        if node not in self.node_list:
            continue
        nodes_in_clusters_dict[cluster].append(node)
    
    real_node_delay_ts = np.zeros((real_cluster_timeseries.shape[0], len(self.node_list)))
    for cluster_label in self.cluster_list:
        cluster_nodes = nodes_in_clusters_dict[cluster_label]
        index_cluster = self.cluster_list.index(cluster_label)
        cluster_delay_ts = real_cluster_timeseries[:, index_cluster]
        for v in cluster_nodes:
            index_v = self.node_list.index(v)
            real_node_delay_ts[:, index_v] = cluster_delay_ts/len(cluster_nodes)
    return real_node_delay_ts

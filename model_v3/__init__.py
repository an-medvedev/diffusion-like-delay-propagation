import json
import numpy as np
import networkx as nx
from datetime import datetime, timedelta
from tqdm.notebook import tqdm
from copy import deepcopy
from collections import defaultdict
from itertools import cycle
import pickle
import os
import re
import warnings
from scipy.signal import savgol_filter

from .utils import *


# DEFAULT CONSTANTS

DAYS_OF_THE_WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DEFAULT_FORBIDDEN_TRAINS = ["EURST", "TGV", "THAL", "IZY", "INT", "EXTRA", "ICE", "CHARTER"]

### NODE BASED MODEL

class PrepareNetworkNodeBasedModel(object):

    def __init__(self, original_network_path, DAT, forbidden_trains = 'default', is_toy_example = False, 
                 toy_network = None):
        ### UPLOAD THE RAILWAY NETWORK AS A GRAPH
        if is_toy_example:
            self.g_delays = toy_network.to_directed()
        else:
            g_train = nx.read_graphml(original_network_path)
            nodelist = list(g_train.nodes)
            self.g_delays = g_train.to_directed()

        # CREATE THE LIST OF NETWORKS FOR EACH DAY OF THE WEEK
        
        if is_toy_example:
            self.H = deepcopy(self.g_delays)
        else:
            self.h_list = []
            for _ in range(7):
                self.h_list.append(deepcopy(self.g_delays))

        # STORE LOCATIONS OF DATA FILES
        if not is_toy_example:
            date_pattern = r"\d{4}-\d{2}-\d{2}"
            files_list = [f for f in sorted(os.listdir(DAT)) if re.search(date_pattern, f) is not None]
            self.path_list = [DAT + f for f in files_list]
            print(f"Data files to read: {len(files_list)}")

        # FORBIDDEN TRAINS NAMES
        if isinstance(forbidden_trains, str):
            self.forbidden_trains = DEFAULT_FORBIDDEN_TRAINS
        elif isinstance(forbidden_trains, list):
            self.forbidden_trains = forbidden_trains
        else:
            print(forbidden_trains)
            raise Exception("forbidden_trains must be either a string or a list")

        print("""Ready to go through the data for the node based model :)
            Use .prepare_network_file to process the data into the raw counts for the model
            Use .prepare_custom_network_file to work with toy examples (remember to set is_toy_example=True)""")

    # IMPORTED METHODS
    from ._network_preparation import prepare_network_file_node_based as prepare_network_file
    from ._network_preparation import prepare_network_custom_node_based as prepare_custom_network_file 
    from ._network_preparation import save_graph_list


class NodeBasedModel(object):
    def __init__(self, model_graph_list_path, forbidden_trains = 'default', verbose = True,
        is_toy_example = False, toy_network = None):
        # UPLOAD PREPARED NETWORK FROM PICKLED FILE
        self.graphs_list = pickle.load(open(model_graph_list_path, "rb"))
        print("""Use .graphs_list to access the networks with raw node metadata for each day of the week""")

        # FORBIDDEN TRAINS NAMES
        if isinstance(forbidden_trains, str):
            self.forbidden_trains = DEFAULT_FORBIDDEN_TRAINS
        elif isinstance(forbidden_trains, list):
            self.forbidden_trains = forbidden_trains
        else:
            print(forbidden_trains)
            raise Exception("forbidden_trains must be either a string or a list")

        self.G = None
        self.B_vector = None
        self.start_day_dt = None
        self.current_date = None
        self.steps = None
        self.is_global_verbose = verbose 

    #Imported methods
    from ._data import store_data_from_directory, upload_raw_data, upload_custom_train_tracks
    from ._data import parse_raw_data_node_based_model as parse_raw_data
    from ._model_preparation import calculate_node_based_model_parameters as calculate_model_parameters


class ClusteredNodeBasedModel():
    def __init__(self, graph_list_path, forbidden_trains = 'default', verbose = True):
        # UPLOAD PREPARED NETWORK FROM PICKLED FILE
        self.graphs_list = pickle.load(open(graph_list_path, "rb"))
        self.G = None
        self.B_vector = None
        if verbose:
            print("""Use .graphs_list to access the networks with raw node metadata for each day of the week""")

        # FORBIDDEN TRAINS NAMES
        if isinstance(forbidden_trains, str):
            self.forbidden_trains = DEFAULT_FORBIDDEN_TRAINS
        elif isinstance(forbidden_trains, list):
            self.forbidden_trains = forbidden_trains
        else:
            print(forbidden_trains)
            raise Exception("forbidden_trains must be either a string or a list")

        self.start_day_dt = None
        self.steps = None
        self.is_global_verbose = verbose


    #Imported methods
    from ._data import store_data_from_directory, upload_raw_data, upload_custom_train_tracks
    from ._data import parse_raw_data_clustered_node_based_model as parse_raw_data
    from ._model_preparation import calculate_clustered_node_based_model_parameters as calculate_model_parameters
    from ._finishing import redistribute_ts_from_clusters_to_nodes, redistribute_real_ts_from_clusters_to_nodes

### EDGE BASED MODEL

class PrepareNetworkEdgeBasedModel(object):
    def __init__(self, original_network_path, data_dir, forbidden_trains = 'default', is_toy_example = False, 
        toy_network = None):
        ### UPLOAD THE RAILWAY NETWORK AS A GRAPH
        if is_toy_example:
            self.g_delays = toy_network.to_directed()
        else:
            g_train = nx.read_graphml(original_network_path)
            nodelist = list(g_train.nodes)
            self.g_delays = g_train.to_directed()

        # CREATE THE LIST OF NETWORKS FOR EACH DAY OF THE WEEK
        # BUILD THE MEMORY GRAPH - THE GRAPH OF EDGE ADJACENCY (AKA THE LINK GRAPH)
        H = nx.DiGraph(name = "memory_graph")            
        for u, data in self.g_delays.nodes(data = True):
            incoming_edges = list(self.g_delays.in_edges(u))
            outgoing_edges = list(self.g_delays.out_edges(u))
            
            for e_in in incoming_edges:
                for e_out in outgoing_edges:
                    if not H.has_node(e_in):
                        H.add_node(e_in)
                    if not H.has_node(e_out):
                        H.add_node(e_out)
                    if not H.has_edge(e_in, e_out):
                        H.add_edge(e_in, e_out)

        # WE COMPUTE PARAMETERS FOR EACH DAY OF THE WEEK SEPARATELY
        # CREATE THE LIST OF NETWORKS FOR EACH DAY OF THE WEEK

        if is_toy_example:
            self.H = deepcopy(H)
        else:
            self.h_list = []
            for _ in range(7):
                self.h_list.append(deepcopy(H))

        # STORE LOCATIONS OF DATA FILES
        if not is_toy_example:
            date_pattern = r"\d{4}-\d{2}-\d{2}"
            files_list = [f for f in sorted(os.listdir(data_dir)) if re.search(date_pattern, f) is not None]
            data_path_list = [data_dir + f for f in files_list]
            print(f"Data files to read: {len(files_list)}")

        # FORBIDDEN TRAINS NAMES
        if isinstance(forbidden_trains, str):
            self.forbidden_trains = DEFAULT_FORBIDDEN_TRAINS
        elif isinstance(forbidden_trains, list):
            self.forbidden_trains = forbidden_trains
        else:
            raise Exception("forbidden_trains must be either a string or a list")

        print("""Ready to go through the data for the node based model :)
            Use .prepare_network_file to process the data into the raw counts for the model
            Use .prepare_custom_network_file to work with toy examples (remember to set is_toy_example=True)""")

    # IMPORTED METHODS
    from ._network_preparation import prepare_network_file_edge_based as prepare_network_file
    from ._network_preparation import prepare_network_custom_edge_based as prepare_custom_network_file 
    from ._network_preparation import save_graph_list

class EdgeBasedModel(object):
    def __init__(self, original_network_path, model_graph_list_path, forbidden_trains = 'default', verbose = True, 
        is_toy_example = False, toy_network = None):
        # UPLOAD PREPARED NETWORK FROM PICKLED FILE
        self.graphs_list = pickle.load(open(model_graph_list_path, "rb"))
        print("""Use .graphs_list to access the networks with raw edge metadata for each day of the week""")

        ### UPLOAD THE RAILWAY NETWORK AS A GRAPH
        if is_toy_example:
            self.g = toy_network
        else:
            g_train = nx.read_graphml(original_network_path)
            nodelist = list(g_train.nodes)
            self.g = g_train.to_directed()

        # FORBIDDEN TRAINS NAMES
        if isinstance(forbidden_trains, str):
            self.forbidden_trains = DEFAULT_FORBIDDEN_TRAINS
        elif isinstance(forbidden_trains, list):
            self.forbidden_trains = forbidden_trains
        else:
            print(forbidden_trains)
            raise Exception("forbidden_trains must be either a string or a list")

        self.G = None
        self.B_vector = None
        self.start_day_dt = None
        self.steps = None
        self.current_date = None
        self.is_global_verbose = verbose

    #Imported methods
    from ._data import store_data_from_directory, upload_raw_data, upload_custom_train_tracks
    from ._data import parse_raw_data_edge_based_model as parse_raw_data
    from ._model_preparation import calculate_edge_based_model_parameters as calculate_model_parameters
    from ._finishing import get_node_delay_aggregation



class ClusteredEdgeBasedModel(object):
    def __init__(self, original_network_path, model_graph_list_path, forbidden_trains = 'default', verbose = True):
        # UPLOAD PREPARED NETWORK FROM PICKLED FILE
        self.graphs_list = pickle.load(open(model_graph_list_path, "rb"))
        self.G = None
        self.B_vector = None
        print("""Use .graphs_list to access the networks with raw node metadata for each day of the week""")

        ### UPLOAD THE RAILWAY NETWORK AS A GRAPH - WE NEED IT TO CORRECT THE TRACKS
        g_train = nx.read_graphml(original_network_path)
        nodelist = list(g_train.nodes)
        self.g = g_train.to_directed()

        # FORBIDDEN TRAINS NAMES
        if isinstance(forbidden_trains, str):
            self.forbidden_trains = DEFAULT_FORBIDDEN_TRAINS
        elif isinstance(forbidden_trains, list):
            self.forbidden_trains = forbidden_trains
        else:
            print(forbidden_trains)
            raise Exception("forbidden_trains must be either a string or a list")

        self.start_day_dt = None
        self.steps = None
        self.is_global_verbose = verbose

    #Imported methods
    from ._data import store_data_from_directory, upload_raw_data, upload_custom_train_tracks
    from ._data import parse_raw_data_clustered_edge_based as parse_raw_data
    from ._model_preparation import calculate_clustered_edge_based_model_parameters as calculate_model_parameters
    from ._finishing import get_node_cluster_delay_aggregation

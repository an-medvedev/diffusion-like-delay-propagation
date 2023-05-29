import numpy as np
from datetime import datetime, timedelta

from .utils import *

# NODE BASED MODEL

def calculate_node_based_model_parameters(self, g):
    NUM_PERIODS = g.graph["num_periods"]
    N = len(g)                               # number of edges in the railway network
    dp = get_period_of_day(NUM_PERIODS)      # dictionary to map hours to day periods
    
    # MATRIX B 
    B_vector = np.zeros((NUM_PERIODS, N))   # 1/avg traversal - rate of delay propagation
    for index_hour in tqdm(range(NUM_PERIODS), disable = True):
        for j, (u, data) in enumerate(g.nodes(data = True)):
            sum_frequency = sum([edge_data["num_trains_continue"][index_hour]  
                                 for u_out, u_in, edge_data in g.in_edges(u, data = True) 
                                 if "num_trains_continue" in edge_data])

            sum_frequency_traversal = sum([(edge_data["num_trains_continue"][index_hour] * 
                                            edge_data["traversal_time"][index_hour]) 
                                           for u_out, u_in, edge_data in g.in_edges(u, data = True) 
                                           if "num_trains_continue" in edge_data])
            if sum_frequency_traversal != 0:
                B_vector[index_hour][j] = sum_frequency/sum_frequency_traversal
            else:
                B_vector[index_hour][j] = 0.

    # PROBABILITY TO CONTINUE
    P_matrix = np.zeros((NUM_PERIODS, N, N))
    for index_hour in tqdm(range(NUM_PERIODS), disable = True):
        for u, v, data in g.edges(data = True):
            # COMPUTE THE PROBABILITY TO STOP THE TRAIN  
            node_data = g.nodes[u]
            if "stopped_trains" in node_data and "passage_count" in node_data:
                all_passing_trains = node_data["passage_count"][index_hour]
                if all_passing_trains != 0:
                    stop_probability = node_data["stopped_trains"][index_hour]/all_passing_trains
                else:
                    stop_probability = 0.

            else:
                stop_probability = 0

            # IF TRAIN DOESN'T STOP, THEN WHERE IT PROPORTIONALLY GO
            if "num_trains_continue" in data:
                freq_ij = data["num_trains_continue"][index_hour]
                sum_out_freq = sum([edge_data["num_trains_continue"][index_hour]  
                                    for u_out, u_in, edge_data in g.out_edges(u, data = True)
                                    if "num_trains_continue" in edge_data])
                if sum_out_freq != 0:
                    r_ij = freq_ij/sum_out_freq
                else:
                    r_ij = 0
            else:
                r_ij = 0.

            index_u = list(g.nodes()).index(u)
            index_v = list(g.nodes()).index(v)
            P_matrix[index_hour][index_u][index_v] = r_ij*(1.-stop_probability)

    # TOTAL MATRIX OF PARAMETERS G

    G = np.zeros((NUM_PERIODS, N, N))
    for index_hour in range(NUM_PERIODS):
        # WE PROPAGATE THE VECTOR B TO MATRIX BY CLONING IT N TIMES AND MAKING A COLUMN FROM IT
        # BELOW IS THE EXAMPLE OF HOW IT WORKS

    #     A = np.array([[1,2,3], [4,5,6]])
    #     x = np.array([2,3])
    #     x_mega = np.array([x,]*3).transpose()
    #     np.multiply(A, x_mega)
        G[index_hour] = np.multiply(P_matrix[index_hour], np.array([B_vector[index_hour],]*N).T).T

    self.G = G
    self.B_vector = B_vector
    self.g = g
    print("Model parameters are ready. You can start the simulation.")



def calculate_clustered_node_based_model_parameters(self, g, cluster_dict):
    # RECORD THE CLUSTER DICTIONARY TO USE IT FURTHER
    self.cluster_dict = cluster_dict

    NUM_PERIODS = g.graph["num_periods"]
    N = len(g)                               # number of edges in the railway network
    dp = get_period_of_day(NUM_PERIODS)      # dictionary to map hours to day periods

    # CREATE THE LIST OF NEW NETWORKS OF CLUSTERS FOR EACH PERIOD
    self.cluster_list = sorted(list(set(self.cluster_dict.values())))
    nodes_in_clusters_dict = defaultdict(list)
    for node, cluster in self.cluster_dict.items():
        if node not in g.nodes:
            continue
        nodes_in_clusters_dict[cluster].append(node)

    # INITIALISE THE NETWORK
    g_clustered = nx.DiGraph(name = "Clustered network")

    # ADD NODES TO THE NETWORK
    for c1 in self.cluster_list:
        nodes_in_cluster = nodes_in_clusters_dict[c1]
        if nodes_in_cluster != 0:
            g_clustered.add_node(c1, stopping_probability = np.zeros(NUM_PERIODS))
            
            # CALCULATE THE STOPPING PROBABILITY ACCORDING TO THE TEXT WITH Q AND S AS IN TEXT
            Q = []
            sum_incoming_frequency = np.zeros(NUM_PERIODS)
            for u_c1 in nodes_in_cluster:
                for u, v, data in g.in_edges(u_c1, data = True):
                    sum_incoming_frequency += np.array(data["num_trains_continue"])

            for u_c1 in nodes_in_cluster:
                incoming_frequency = np.zeros(NUM_PERIODS)
                for u, v, data in g.in_edges(u_c1, data = True):
                     incoming_frequency += np.array(data["num_trains_continue"])
               
                q_array = np.zeros(NUM_PERIODS)
                for index_hour, (in_freq, sum_in_freq) in enumerate(zip(incoming_frequency, sum_incoming_frequency)):
                    if sum_in_freq != 0:
                        q_array[index_hour] = in_freq/sum_in_freq
                Q.append(q_array)

            S = []  
            for u_c1 in nodes_in_cluster:
                node_data = g.nodes[u_c1]
                stop_probability = np.zeros(NUM_PERIODS)
                if "stopped_trains" in node_data and "passage_count" in node_data:
                    stopped_trains = node_data["stopped_trains"]
                    all_passing_trains = node_data["passage_count"]
                    for index_hour, (st_trains, all_st_trains) in enumerate(zip(stopped_trains, all_passing_trains)):
                        if all_st_trains != 0:
                            stop_probability[index_hour] = st_trains/all_st_trains

                S.append(stop_probability)

            g_clustered.nodes[c1]["Q"] = Q
            g_clustered.nodes[c1]["S"] = S
            g_clustered.nodes[c1]["stopping_probability"] = np.sum(np.array(S)*np.array(Q), axis = 0)

    # ADD EDGES TO THE NETWORK
    for u1, u2 in g.edges:
        # PREPARE THE VARIABLES TO CALCULATE THE NEW F_IJ AND T_IJ
        if u1 in cluster_dict and u2 in cluster_dict:
            c_u1 = cluster_dict[u1]
            c_u2 = cluster_dict[u2]
            if not g_clustered.has_edge(c_u1, c_u2):
                g_clustered.add_edge(c_u1, c_u2, num_trains_continue = np.zeros(NUM_PERIODS),
                                     num_trains_traversal_time = np.zeros(NUM_PERIODS),
                                     traversal_time = np.zeros(NUM_PERIODS))

            g_clustered[c_u1][c_u2]["num_trains_continue"] += np.array(g[u1][u2]["num_trains_continue"])
            g_clustered[c_u1][c_u2]["num_trains_traversal_time"] += np.array(g[u1][u2]["num_trains_continue"])*np.array(g[u1][u2]["traversal_time"])

    # CALCULATE THE NEW T_IJ
    for c1, c2 in g_clustered.edges:
        for index_hour, (sum_traversal_time, sum_frequency) in enumerate(zip(g_clustered[c1][c2]["num_trains_traversal_time"], g_clustered[c1][c2]["num_trains_continue"])):
            if sum_frequency != 0:
                g_clustered[c1][c2]["traversal_time"][index_hour] = sum_traversal_time/sum_frequency

    N_clusters = len(self.cluster_list)

    # MATRIX B 
    B_vector = np.zeros((NUM_PERIODS, N_clusters)) 
    for index_hour in tqdm(range(NUM_PERIODS), disable = True):
        for u, data in g_clustered.nodes(data = True):
            sum_frequency = sum([edge_data["num_trains_continue"][index_hour]  
                                 for u_out, u_in, edge_data in g_clustered.in_edges(u, data = True) 
                                 if "num_trains_continue" in edge_data])

            sum_frequency_traversal = sum([(edge_data["num_trains_continue"][index_hour] * 
                                            edge_data["traversal_time"][index_hour]) 
                                           for u_out, u_in, edge_data in g_clustered.in_edges(u, data = True) 
                                           if "num_trains_continue" in edge_data])
            index_u = self.cluster_list.index(u)
            if sum_frequency_traversal != 0:
                B_vector[index_hour][index_u] = sum_frequency/sum_frequency_traversal
            else:
                B_vector[index_hour][index_u] = 0.

    # PROBABILITY TO CONTINUE
    P_matrix = np.zeros((NUM_PERIODS, N_clusters, N_clusters))
    for index_hour in tqdm(range(NUM_PERIODS), disable = True):
        for c1, c2, data in g_clustered.edges(data = True):
            # GET THE STOPPING PROBABILITY
            if "stopping_probability" in g_clustered.nodes[c1]:
                stop_probability = g_clustered.nodes[c1]["stopping_probability"][index_hour]
            else:
                stop_probability = 0.

            # IF TRAIN DOESN'T STOP, THEN WHERE IT PROPORTIONALLY GO
            if "num_trains_continue" in data:
                freq_ij = data["num_trains_continue"][index_hour]
                sum_out_freq = sum([edge_data["num_trains_continue"][index_hour]  
                                    for _, _, edge_data in g_clustered.out_edges(c1, data = True)
                                    if "num_trains_continue" in edge_data])
                if sum_out_freq != 0:
                    r_ij = freq_ij/sum_out_freq
                else:
                    r_ij = 0.
            else:
                r_ij = 0.

            index_c1 = list(self.cluster_list).index(c1)
            index_c2 = list(self.cluster_list).index(c2)
            P_matrix[index_hour][index_c1][index_c2] = r_ij*(1.-stop_probability)

    # TOTAL MATRIX OF PARAMETERS G

    G = np.zeros((NUM_PERIODS, N_clusters, N_clusters))
    for index_hour in range(NUM_PERIODS):
        # WE PROPAGATE THE VECTOR B TO MATRIX BY CLONING IT N TIMES AND MAKING A COLUMN FROM IT
        # BELOW IS THE EXAMPLE OF HOW IT WORKS

    #     A = np.array([[1,2,3], [4,5,6]])
    #     x = np.array([2,3])
    #     x_mega = np.array([x,]*3).transpose()
    #     np.multiply(A, x_mega)
        G[index_hour] = np.multiply(P_matrix[index_hour], 
                            np.array([B_vector[index_hour],]*N_clusters).T).T 

    self.P_matrix = P_matrix
    self.B_vector = B_vector
    self.g_clustered = g_clustered
    self.g = g
    self.G = G
    if self.is_global_verbose:
        print("""Model parameters are ready. You can start the simulation.
            Use .g_clustered to access the parameters of the cluster aggregated network""")



# EDGE BASED MODEL

def calculate_edge_based_model_parameters(self, h):
    NUM_PERIODS = h.graph["num_periods"]
    M = len(h)                               # number of edges in the railway network
    dp = get_period_of_day(NUM_PERIODS)      # dictionary to map hours to day periods
    
    # MATRIX B 
    T_vector = np.zeros((NUM_PERIODS, M))   # Average traversal time
    B_vector = np.zeros((NUM_PERIODS, M))   # 1/avg traversal - rate of delay propagation
    no_traversal_time = []
    for index_hour in tqdm(range(NUM_PERIODS), disable = True):
        for j, (u, data) in enumerate(h.nodes(data = True)):
            if "traversal_time" in data:
                avg_traversal_time = data["traversal_time"][index_hour] 
                T_vector[index_hour][j] = avg_traversal_time
                if avg_traversal_time != 0:
                    B_vector[index_hour][j] = 1/avg_traversal_time
                else:
                    B_vector[index_hour][j] = 0.
            else:
                no_traversal_time.append(u)
                B_vector[index_hour][j] = 0.

    # PROBABILITY TO STOP
    ST_vector = np.zeros((NUM_PERIODS, M))    # stopping probability at edges (at an end station of and edge)
    for index_hour in tqdm(range(NUM_PERIODS), disable = True):
        for j, (u, data) in enumerate(h.nodes(data = True)):
            if "stopped_trains" in data:
                all_passing_trains = data["passing_trains"][index_hour]
                if all_passing_trains != 0:
                    stop_probability = data["stopped_trains"][index_hour]/all_passing_trains
                else:
                    stop_probability = 0.
            else:
                stop_probability = 0

            ST_vector[index_hour][j] = stop_probability

    # PROBABILITY TO TRAVERSE FROM EDGE TO EDGE
    P_matrix = np.zeros((NUM_PERIODS, M, M))   # main matrix of diffusion redistribution
    for index_hour in tqdm(range(NUM_PERIODS), disable = True):
        for ei, ej, data in h.edges(data = True):
            freq_ei_ej = data["traversal_btw_edges_count"][index_hour] 
            sum_out_freq = 0
            for ek in h.successors(ei):
                sum_out_freq += h[ei][ek]["traversal_btw_edges_count"][index_hour]
            if sum_out_freq != 0:
                if freq_ei_ej == sum_out_freq:
                    r_ei_ej = 1.
                else:
                    r_ei_ej = freq_ei_ej/sum_out_freq
            else:
                r_ei_ej = 0

            index_ei = list(h.nodes()).index(ei)
            index_ej = list(h.nodes()).index(ej)  

            P_matrix[index_hour][index_ei][index_ej] = r_ei_ej*(1. - ST_vector[index_hour][index_ei])
            
    # INFLUX OPERATOR
    G = np.zeros((NUM_PERIODS, M, M))
    for index_hour in tqdm(range(NUM_PERIODS), disable = True):
        # WE PROPAGATE THE VECTOR B TO MATRIX BY CLONING IT N TIMES AND MAKING A COLUMN FROM IT
        # BELOW IS THE EXAMPLE OF HOW IT WORKS
        #     A = np.array([[1,2,3], [4,5,6]])
        #     x = np.array([2,3])
        #     x_mega = np.array([x,]*3).transpose()
        #     np.multiply(A, x_mega)
        G[index_hour] = np.multiply(P_matrix[index_hour], np.array([B_vector[index_hour],]*M).T).T
    
    self.G = G
    self.B_vector = B_vector
    self.h = h
    if self.is_global_verbose:
        print("Model parameters are ready. You can start the simulation.")


def calculate_clustered_edge_based_model_parameters(self, h, cluster_dict):
    # RECORD THE CLUSTER DICTIONARY TO USE IT FURTHER
    self.cluster_dict = cluster_dict

    NUM_PERIODS = h.graph["num_periods"]
    M = len(h)                               # number of edges in the railway network
    dp = get_period_of_day(NUM_PERIODS)      # dictionary to map hours to day periods

    # CREATE THE SET (UNORDERED) OF ALL NODES IN NETWORK
    set_nodes_in_network = set([])
    for (e1, e2) in h.nodes:
        set_nodes_in_network.add(e1)
        set_nodes_in_network.add(e2)

    # CREATE THE LIST OF NEW NETWORKS OF CLUSTERS FOR EACH PERIOD
    self.cluster_list = sorted(list(set(self.cluster_dict.values())))
    nodes_in_clusters_dict = defaultdict(list)
    for node, cluster in self.cluster_dict.items():
        if node not in set_nodes_in_network:
            continue
        nodes_in_clusters_dict[cluster].append(node)

    edges_in_cluster_dict = defaultdict(list)
    for e1, e2 in h.nodes:
        cluster_e2 = cluster_dict[e2]
        edges_in_cluster_dict[cluster_e2].append((e1,e2))

    # ANALOGUE OF CLUSTER_DICT FOR RAILWAY NETWORK EDGES
    edge_cluster_dict = {}
    for cluster, edges_in_cluster in edges_in_cluster_dict.items():
        for e in edges_in_cluster:
            edge_cluster_dict[e] = cluster

    # INITIALISE THE CLUSTERED NETWORK
    h_clustered = nx.DiGraph(name = "Clustered network")

    # ADD NODES TO THE CLUSTERED NETWORK
    for (u, v) in h.nodes:
        c_u = self.cluster_dict[u]
        c_v = self.cluster_dict[v]
        c_edge = (c_u, c_v)
        if not h_clustered.has_node(c_edge):
            h_clustered.add_node(c_edge, stopped_trains = np.zeros(NUM_PERIODS),
                                    passing_trains = np.zeros(NUM_PERIODS),
                                    num_trains_times_traversal_time = np.zeros(NUM_PERIODS),
                                    traversal_time = np.zeros(NUM_PERIODS))
        if "passing_trains" in h.nodes[(u, v)]:
            h_clustered.nodes[c_edge]["passing_trains"] += np.array(h.nodes[(u, v)]["passing_trains"])
            if "traversal_time" in h.nodes[(u, v)]:
                h_clustered.nodes[c_edge]["num_trains_times_traversal_time"] += np.array(h.nodes[(u, v)]["passing_trains"])*np.array(h.nodes[(u, v)]["traversal_time"])
        if "stopped_trains" in h.nodes[(u, v)]:
            h_clustered.nodes[c_edge]["stopped_trains"] += np.array(h.nodes[(u, v)]["stopped_trains"])
    
    for c_edge in h_clustered.nodes:
        for index_hour, (num_trains_times_traversal, num_passing_trains) in enumerate(zip(h_clustered.nodes[c_edge]["num_trains_times_traversal_time"], h_clustered.nodes[c_edge]["passing_trains"])):
            if num_passing_trains != 0:
                h_clustered.nodes[c_edge]["traversal_time"][index_hour] = num_trains_times_traversal/num_passing_trains

    # ADD EDGES TO THE CLUSTERED NETWORK
    for (e1, e2) in h.edges:
        # CALCULATE THE SUM OF ALL TRAINS ON EDGES AND PREPARE FOR THE COMPUTATION OF AVERAGE TRAVERSAL TIME
        c_e11, c_e12 = cluster_dict[e1[0]], cluster_dict[e1[1]]
        c_e21, c_e22 = cluster_dict[e2[0]], cluster_dict[e2[1]]

        if not h_clustered.has_edge((c_e11, c_e12), (c_e21, c_e22)):
            h_clustered.add_edge((c_e11, c_e12), (c_e21, c_e22), traversal_btw_edges_count = np.zeros(NUM_PERIODS))
        
        h_clustered[(c_e11, c_e12)][(c_e21, c_e22)]["traversal_btw_edges_count"] += np.array(h[e1][e2]["traversal_btw_edges_count"])
        
    self.h_clustered = h_clustered
    M_clusters = len(h_clustered)
    
    # MATRIX B 
    T_vector = np.zeros((NUM_PERIODS, M_clusters))   # Average traversal time
    B_vector = np.zeros((NUM_PERIODS, M_clusters))   # 1/avg traversal - rate of delay propagation
    no_traversal_time = []
    for index_hour in tqdm(range(NUM_PERIODS), disable = True): 
        for j, (u, data) in enumerate(h_clustered.nodes(data = True)):
            if "traversal_time" in data:
                avg_traversal_time = data["traversal_time"][index_hour] 
                T_vector[index_hour][j] = avg_traversal_time
                if avg_traversal_time != 0:
                    B_vector[index_hour][j] = 1/avg_traversal_time
                else:
                    B_vector[index_hour][j] = 0.
            else:
                no_traversal_time.append(u)
                B_vector[index_hour][j] = 0.

    # PROBABILITY TO STOP
    ST_vector = np.zeros((NUM_PERIODS, M_clusters))    # stopping probability at edges (at an end station of and edge)
    for index_hour in tqdm(range(NUM_PERIODS), disable = True):
        for j, (u, data) in enumerate(h_clustered.nodes(data = True)):
            if "stopped_trains" in data:
                all_passing_trains = data["passing_trains"][index_hour]
                if all_passing_trains != 0:
                    stop_probability = data["stopped_trains"][index_hour]/all_passing_trains
                else:
                    stop_probability = 0.
            else:
                stop_probability = 0

            ST_vector[index_hour][j] = stop_probability

    # PROBABILITY TO TRAVERSE FROM EDGE TO EDGE
    P_matrix = np.zeros((NUM_PERIODS, M_clusters, M_clusters))   # main matrix of diffusion redistribution
    for index_hour in tqdm(range(NUM_PERIODS), disable = True):
        for ei, ej, data in h_clustered.edges(data = True):
            freq_ei_ej = data["traversal_btw_edges_count"][index_hour] 
            sum_out_freq = 0
            for ek in h_clustered.successors(ei):
                sum_out_freq += h_clustered[ei][ek]["traversal_btw_edges_count"][index_hour]
            if sum_out_freq != 0:
                if freq_ei_ej == sum_out_freq:
                    r_ei_ej = 1.
                else:
                    r_ei_ej = freq_ei_ej/sum_out_freq
            else:
                r_ei_ej = 0

            index_ei = list(h_clustered.nodes).index(ei)
            index_ej = list(h_clustered.nodes).index(ej)  

            P_matrix[index_hour][index_ei][index_ej] = r_ei_ej*(1. - ST_vector[index_hour][index_ei])
            
    # INFLUX OPERATOR
    G = np.zeros((NUM_PERIODS, M_clusters, M_clusters))
    for index_hour in tqdm(range(NUM_PERIODS), disable = True):
        # WE PROPAGATE THE VECTOR B TO MATRIX BY CLONING IT N TIMES AND MAKING A COLUMN FROM IT
        # BELOW IS THE EXAMPLE OF HOW IT WORKS
        #     A = np.array([[1,2,3], [4,5,6]])
        #     x = np.array([2,3])
        #     x_mega = np.array([x,]*3).transpose()
        #     np.multiply(A, x_mega)
        G[index_hour] = np.multiply(P_matrix[index_hour], np.array([B_vector[index_hour],]*M_clusters).T).T
    
    self.G = G
    self.P_matrix = P_matrix
    self.B_vector = B_vector
    
    self.h = h
    if self.is_global_verbose:
        print("Model parameters are ready. You can start the simulation.")

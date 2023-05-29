# import json
# import numpy as np
# import networkx as nx
# from datetime import datetime, timedelta
# from tqdm.notebook import tqdm
# from copy import deepcopy
# from collections import defaultdict
# from itertools import cycle
# import os

from .utils import * 

### TECHNICAL FUNCTIONS

# NODE-BASED MODEL
def add_stop_trains_graph(G, s, index_period, num_periods):
    # THIS FUNCTION ADDS ONE TO THE COUNTER OF STOPPED TRAINS AT SPECIFIED HOUR
    # IF THE COUNTER DOESN'T EXIST - IT CREATES A NEW ONE
    # RETURNS GRAPH
    if G.has_node(s):
        if "stopped_trains" in G.nodes[s]:
            G.nodes[s]["stopped_trains"][index_period] += 1
        else:
            stopped_trains_list = list(np.zeros(num_periods))
            stopped_trains_list[index_period] += 1
            G.nodes[s]["stopped_trains"] = stopped_trains_list
    return G

def add_passage_count_graph(G, s, index_period, num_periods):
    # THIS FUNCTION ADDS PASSAGE COUNT TO THE LIST CONTAINER OF A NODE
    # IF THE PASSAGE COUNT LIST DOESN'T EXIST - IT CREATES A NEW ONE
    # RETURNS GRAPH
    if G.has_node(s):
        if "passage_count" in G.nodes[s]:
            G.nodes[s]["passage_count"][index_period] += 1
        else:
            passage_count_list = list(np.zeros(num_periods))
            passage_count_list[index_period] += 1
            G.nodes[s]["passage_count"] = passage_count_list
    return G

def add_traversal_time_graph(G, s_out, s_in, delta_time, index_period, num_periods):
    # THIS FUNCTION ADDS TRAVERSAL TIME TO THE LIST CONTAINER OF AN EDGE IN THE GRAPH
    # IF THE TRAVERSAL TIME LIST DOESN'T EXIST - IT CREATES A NEW ONE
    # RETURNS GRAPH
    if G.has_edge(s_out, s_in):
        if "traversal_time" in G[s_out][s_in]:
            G[s_out][s_in]["traversal_time"][index_period].append(delta_time)
        else:
            traversal_time = [[] for _ in range(num_periods)]
            traversal_time[index_period].append(delta_time)
            G[s_out][s_in]["traversal_time"] = traversal_time
    return G

def add_num_trains_continue_graph(G, s_out, s_in, index_period, num_periods):
    # THIS FUNCTION ADDS TRAVERSAL COUNT TO THE LIST CONTAINER OF AN EDGE
    # IF THE TRAVERSAL COUNT LIST DOESN'T EXIST - IT CREATES A NEW ONE
    # RETURNS GRAPH
    if G.has_edge(s_out, s_in):
        if "num_trains_continue" in G[s_out][s_in]:
            G[s_out][s_in]["num_trains_continue"][index_period] += 1
        else:
            traversal_direction_count_list = list(np.zeros(num_periods))
            traversal_direction_count_list[index_period] += 1
            G[s_out][s_in]["num_trains_continue"] = traversal_direction_count_list
    return G




# EDGE-BASED MODEL
def add_stop_trains_to_edges(G, s_out, s_in, index_period, num_periods):
    # THIS FUNCTION ADDS ONE TO THE COUNTER OF STOPPED TRAINS AT SPECIFIED HOUR
    # IF THE COUNTER DOESN'T EXIST - IT CREATES A NEW ONE
    # RETURNS GRAPH
    if G.has_node((s_out, s_in)):
        if "stopped_trains" in G.nodes[(s_out, s_in)]:
            G.nodes[(s_out, s_in)]["stopped_trains"][index_period] += 1
        else:
            stopped_trains_list = list(np.zeros(num_periods))
            stopped_trains_list[index_period] += 1
            G.nodes[(s_out, s_in)]["stopped_trains"] = stopped_trains_list
    return G

def add_traversal_time_to_edges(G, s_out, s_in, delta_time, index_period, num_periods):
    # THIS FUNCTION ADDS TRAVERSAL TIME TO THE LIST CONTAINER OF A NODE IN MEMORY GRAPH
    # IF THE TRAVERSAL TIME LIST DOESN'T EXIST - IT CREATES A NEW ONE
    # RETURNS GRAPH
    if G.has_node((s_out, s_in)):
        if "traversal_time" in G.nodes[(s_out, s_in)]:
            G.nodes[(s_out, s_in)]["traversal_time"][index_period].append(delta_time)
        else:
            traversal_time = [[] for _ in range(num_periods)]
            traversal_time[index_period].append(delta_time)
            G.nodes[(s_out, s_in)]["traversal_time"] = traversal_time
    return G

def add_passing_trains_to_edges(G, s_out, s_in, index_period, num_periods):
    # THIS FUNCTION ADDS PASSAGE COUNT TO THE LIST CONTAINER OF AN EDGE
    # IF THE PASSAGE COUNT LIST DOESN'T EXIST - IT CREATES A NEW ONE
    # RETURNS GRAPH
    if G.has_node((s_out, s_in)):
        if "passing_trains" in G.nodes[(s_out, s_in)]:
            G.nodes[(s_out, s_in)]["passing_trains"][index_period] += 1
        else:
            passage_count_list = list(np.zeros(num_periods))
            passage_count_list[index_period] += 1
            G.nodes[(s_out, s_in)]["passing_trains"] = passage_count_list
    return G

def add_passage_trains_between_edges(G, s_out, s_inter, s_in, index_period, num_periods):
    # THIS FUNCTION ADDS TRAVERSAL COUNT TO THE LIST CONTAINER OF AN EDGE
    # IF THE TRAVERSAL COUNT LIST DOESN'T EXIST - IT CREATES A NEW ONE
    # RETURNS GRAPH
    if G.has_edge((s_out, s_inter), (s_inter, s_in)):
        if "traversal_btw_edges_count" in G[(s_out, s_inter)][(s_inter, s_in)]:
            G[(s_out, s_inter)][(s_inter, s_in)]["traversal_btw_edges_count"][index_period] += 1
        else:
            traversal_btw_edges_count_list = list(np.zeros(num_periods))
            traversal_btw_edges_count_list[index_period] += 1
            G[(s_out, s_inter)][(s_inter, s_in)]["traversal_btw_edges_count"] = traversal_btw_edges_count_list
    return G
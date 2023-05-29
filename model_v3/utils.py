import json
import numpy as np
import networkx as nx
from datetime import datetime, timedelta
from tqdm.notebook import tqdm
from copy import deepcopy
from collections import defaultdict
from itertools import cycle

import os

# GENERAL TECHNICAL FUNCTIONS

# return the period of the day (we split days into either 1, 2, 4... -hour periods)
def get_period_of_day(num_periods):
    period_h = {}
    period_length = int(24/num_periods)
    for i in range(24):
        period_h[i] = int(i/period_length)
    return period_h

# returns a list of tuples
def get_date_range(month_start, year_start, month_end, year_end):
    month_range = list(range(1,13))
    cycle_month_range = cycle(month_range)
    while True:
        current_month = next(cycle_month_range)
        if current_month == month_start:
            break
    date_tuples = []
    year = year_start
    while True:
        if current_month < 10:
            date_tuples.append(("0"+str(current_month), str(year)))
        else:
            date_tuples.append((str(current_month), str(year)))
        if year == year_end and current_month == month_end:
            break
        current_month = next(cycle_month_range)
        if current_month == 1:
            year += 1
    return date_tuples

def get_corrected_track(track, G): # minor changes
    # argument is a track from a tracklist from the data, which possibly skips stations 
    # or has stations which are not in the network G
    # remove the stops which are not in the network
    # then, fill everything up using shortest path
    # edgestoskip is a list of edges which should not be 'filled' up
    # return corrected track
    
    shortertrack = []
    # go through stations, keep those which are in the network
    
    for s in track:
        if s[0] in G.nodes():
            shortertrack.append(s)
    
    newtrack = [] # brand new track with added intermediate nodes
    
    for s_out, s_in in zip(shortertrack[:-1], shortertrack[1:]):
        
        if s_out[0] in G.nodes():
            newtrack.append(s_out)
        try:
            sh_path = nx.shortest_path(G, s_out[0], s_in[0])
        except:
            continue
        # for each station in shortestpath, insert this station into newtrack
        L = len(sh_path)
        startdel = s_out[-1]
        enddel = s_in[-2]
        deltadelay = (enddel - startdel) / (L-1) # arr delay - dep delay divided by segments. Integer, seconds
        et_scheduled = s_in[1] - s_out[3] # edge time according to scheduled times, timedelta object
        et_real = s_in[2] - s_out[4] # real edge time. datetime object
        deltaetscheduled = et_scheduled/(L-1) # assumption: every intermediate track takes equal time
        deltaetreal = et_real/(L-1)

        for j,s in enumerate(sh_path[1:-1]): # can be empty
            cdel = startdel + (j+1)*deltadelay # 
            stop = [s, s_out[3] + (j+1)*deltaetscheduled, s_out[4] + (j+1)*deltaetreal, \
                    s_out[3] + (j+1)*deltaetscheduled, s_out[4] + (j+1)*deltaetreal, cdel, cdel] # we do not alter the times, only the delays
            # now insert this into newtrack at the correct positions
            newtrack.append(stop)
    #add final stop
    if shortertrack[-1][0] in G.nodes():
        newtrack.append(shortertrack[-1])
    return newtrack

# loads the tracks from file to dictionary, with all the times converted to datetime objects
def upload_schedule(in_file):
    def date_hook(json_dict):
        for (key, value) in json_dict.items():
            for index_list_entry in range(len(value)):
                for index_entry, v in enumerate(value[index_list_entry]):
                    try:
                        json_dict[key][index_list_entry][index_entry] = datetime.strptime(v, "%Y-%m-%d %H:%M:%S")
                    except:
                        pass
        return json_dict

    with open(in_file, "r") as in_f:
        return json.load(in_f, object_hook=date_hook) # output: dict 
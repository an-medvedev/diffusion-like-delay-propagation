import time
import pickle
import json
import numpy as np
from tqdm.notebook import tqdm
from copy import deepcopy
from collections import defaultdict
import re
import networkx as nx
from datetime import datetime, timedelta

from .utils import *
from .prepare_weights import *

# DEFAULT CONSTANTS

DEFAULT_FORBIDDEN_TRAINS = ["EURST", "TGV", "THAL", "IZY", "INT", "EXTRA", "ICE", "CHARTER"]

# GENERAL FUNCTION

def save_graph_list(self, avg_h_list, file_path):
    pickle.dump(avg_h_list, open(file_path, "wb"))
    print("Graphs for the model saved successfully!")

# NODE BASED MODEL

def prepare_network_file_node_based(self, num_periods, verbose = True):
    # CHOOSE THE NUMBER OF PERIODS YOU WISH TO SPLIT THE DAY 
    # PERIOD IS THE NUMBER OF HOURS, THUS THERE CAN BE 1,2,4,6,12 OR 24 PERIODS WITH APPRORPIATE 

    NUM_PERIODS = num_periods
    dp = get_period_of_day(NUM_PERIODS)
        
    # COUNTER FOR THE DATA FILES CONSIDERED FOR EACH DAY OF THE WEEK
    wday_counter = [0]*7

    # ITERATE OVER THE DAYS, COLLECT THE RAW DATA
    count_problems_corrected_track = defaultdict(float)
    for day_counter, fpath in tqdm(enumerate(self.path_list), total = len(self.path_list), disable = not verbose): 
        current_date = fpath[-10:]
        print(f"Working with date :: {current_date}")
        wday_index = datetime.strptime(current_date, "%Y-%m-%d").weekday()
        wday_counter[wday_index] += 1

        train_tracks = upload_schedule(fpath) #load this day's tracks
        counter_successful_tracks = 0.
        for train, track in tqdm(train_tracks.items(), total = len(train_tracks), disable = not verbose): 
            # iterate over the lines of the file, each line is one track
            if any(s in train for s in self.forbidden_trains):
                continue
            # correct the track
            try:
                corrected_track = get_corrected_track(track, self.g_delays)
                counter_successful_tracks += 1
            except:
                count_problems_corrected_track[current_date] += 1
                continue
            # obtain the delays from the track and the times when the train crossed edges
            if len(corrected_track)>2:
                for i, (s1, s2) in enumerate(zip(corrected_track, corrected_track[1:])):
                    s_out_name = s1[0]
                    s_in_name = s2[0]

                    # we are working with datetime objects
                    planned_out_dep = s1[3]
                    planned_in_dep = s2[3]

                    # IF THE IN STATION IS THE ENDING, THEN ADD TRAVERSAL TIME TO THE (INTER-IN) EDGE
                    # AND ADD THE PASSAGE COUNTER TOO
                    if planned_in_dep is None: 
                        planned_in_arr = s2[1]
                        # ADD TO THE PASSAGE TIME
                        delta_time = (planned_in_arr - planned_out_dep).total_seconds()
                        d_period = dp[planned_in_arr.hour]
                        self.h_list[wday_index] = add_traversal_time_graph(self.h_list[wday_index], s_out_name, 
                                                                      s_in_name, delta_time, d_period, NUM_PERIODS)
                        
                        # ADD TO THE TRAVERSAL COUNTER AT THE EDGE (TRAINS CONTINUE TO s_in)
                        self.h_list[wday_index] = add_num_trains_continue_graph(self.h_list[wday_index], s_out_name, 
                                                                           s_in_name, d_period, NUM_PERIODS)

                        self.h_list[wday_index] = add_passage_count_graph(self.h_list[wday_index], s_out_name,
                                                                     d_period, NUM_PERIODS)
                        # ADD TO THE END NODE PASSAGE COUNTER
                        self.h_list[wday_index] = add_passage_count_graph(self.h_list[wday_index], s_in_name,
                                                                     d_period, NUM_PERIODS)

                        # ADD TO THE STOPPING STATIONS COUNT
                        self.h_list[wday_index] = add_stop_trains_graph(self.h_list[wday_index], s_in_name,
                                                                 d_period, NUM_PERIODS)
                    else:
                        # ADD TO THE NODE PASSAGE COUNTER
                        d_period = dp[planned_in_dep.hour]
                        self.h_list[wday_index] = add_passage_count_graph(self.h_list[wday_index], s_out_name,
                                                                     d_period, NUM_PERIODS)
                        
                        
                        # ADD TO THE TRAVERSAL COUNTER AT THE EDGE (TRAINS CONTINUE TO s_in)
                        self.h_list[wday_index] = add_num_trains_continue_graph(self.h_list[wday_index], s_out_name, 
                                                                           s_in_name, d_period, NUM_PERIODS)
                        # ADD TO THE PASSAGE TIME
                        delta_time = (planned_in_dep - planned_out_dep).total_seconds()
                        d_period = dp[planned_in_dep.hour]
                        self.h_list[wday_index] = add_traversal_time_graph(self.h_list[wday_index], s_out_name, 
                                                                      s_in_name, delta_time, d_period, NUM_PERIODS)

            if len(corrected_track) == 2:
                s1, s2 = corrected_track
                s_out_name = s1[0]
                s_in_name = s2[0]

                planned_out_dep = s1[3]
                planned_in_arr = s2[1]

                # ADD TO THE NODE PASSAGE COUNTER (s_out)
                d_period = dp[planned_out_dep.hour]
                self.h_list[wday_index] = add_passage_count_graph(self.h_list[wday_index], s_out_name,
                                                             d_period, NUM_PERIODS)
                # ADD TO THE NODE PASSAGE COUNTER (s_in)
                d_period = dp[planned_in_arr.hour]
                self.h_list[wday_index] = add_passage_count_graph(self.h_list[wday_index], s_in_name,
                                                             d_period, NUM_PERIODS)

                # ADD TO THE STOPPING STATIONS COUNT
                self.h_list[wday_index] = add_stop_trains_graph(self.h_list[wday_index], s_in_name,
                                                         d_period, NUM_PERIODS)

                # ADD TO THE TRAVERSAL COUNTER AT THE EDGE (TRAINS CONTINUE TO s_in)
                self.h_list[wday_index] = add_num_trains_continue_graph(self.h_list[wday_index], s_out_name, 
                                                                   s_in_name, d_period, NUM_PERIODS)

                # ADD TO THE PASSAGE TIME
                delta_time = (planned_in_arr - planned_out_dep).total_seconds()
                d_period = dp[planned_in_arr.hour]
                self.h_list[wday_index] = add_traversal_time_graph(self.h_list[wday_index], s_out_name, 
                                                              s_in_name, delta_time, d_period, NUM_PERIODS)
        count_problems_corrected_track[current_date] = round(count_problems_corrected_track[current_date]/len(train_tracks), 3)
    print(f"Problems during correcting tracks (fraction of total): {count_problems_corrected_track}")

    # AVERAGE OUT THE PARAMETERS
    days_of_the_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    avg_h_list = []
    for day_index, d_week in tqdm(enumerate(days_of_the_week), total = len(days_of_the_week)):

        h = self.h_list[day_index]
        h_avg = nx.DiGraph(day = days_of_the_week[day_index], num_periods = NUM_PERIODS)

        for u, data in h.nodes(data = True):
            h_avg.add_node(u, lon = data["lon"], lat = data["lat"])
            # STOPPED TRAINS 
            if "stopped_trains" in data:
                avg_stopped_trains = list(np.array(data["stopped_trains"])/wday_counter[day_index])
                h_avg.nodes[u]["stopped_trains"] = avg_stopped_trains
                
            # PASSAGE COUNT
            if "passage_count" in data:
                avg_passage_count = list(np.array(data["passage_count"])/wday_counter[day_index])
                h_avg.nodes[u]["passage_count"] = avg_passage_count
        
        for u, v, data in h.edges(data = True):
            h_avg.add_edge(u,v)
            # TRAVERSAL TIME
            if "traversal_time" in data:
                avg_traversal_time = [np.mean(x) if len(x) else 0 for x in data["traversal_time"]]
                h_avg[u][v]["traversal_time"] = avg_traversal_time
                
            # NUMBER OF TRAINS THAT CONTINUE
            if "num_trains_continue" in data:
                avg_num_trains_continue = list(np.array(data["num_trains_continue"])/wday_counter[day_index])
                h_avg[u][v]["num_trains_continue"] = avg_num_trains_continue

        # REMOVE THE EDGES WHICH WILL NOT BE TRAVERSED
        count_no_data = []
        for u, v, data in h_avg.edges(data = True):
            if len(data) == 0:
                count_no_data.append((u,v))
        h_avg.remove_edges_from(count_no_data)
        
        # REMOVE HANGING NODES FROM THE NETWORK
        nodes_to_remove = []
        for u in h_avg.nodes():
            if h_avg.degree(u) == 0:
                nodes_to_remove.append(u)
        h_avg.remove_nodes_from(nodes_to_remove)
        
        avg_h_list.append(h_avg)
    return avg_h_list

def prepare_network_custom_node_based(self, schedules_dict, verbose = True):
    NUM_PERIODS = 1
    dp = get_period_of_day(NUM_PERIODS)
    # ITERATE OVER THE SCHEDULES 
    for train, track in tqdm(schedules_dict.items()): 
        # iterate over the lines of the file, each line is one track
        # obtain the delays from the track and the times when the train crossed edges
        if len(track)>2:
            for i, (s1, s2) in enumerate(zip(track, track[1:])):
                s_out_name = s1[0]
                s_in_name = s2[0]

                # we are working with datetime objects
                planned_out_dep = s1[3]
                planned_in_dep = s2[3]

                # IF THE IN STATION IS THE ENDING, THEN ADD TRAVERSAL TIME TO THE (INTER-IN) EDGE
                # AND ADD THE PASSAGE COUNTER TOO
                if planned_in_dep is None: 
                    planned_in_arr = s2[1]
                    # ADD TO THE PASSAGE TIME
                    delta_time = (planned_in_arr - planned_out_dep).total_seconds()
                    d_period = dp[planned_in_arr.hour]
                    self.H = add_traversal_time_graph(self.H, s_out_name, 
                                                                  s_in_name, delta_time, d_period, NUM_PERIODS)
                    
                    # ADD TO THE TRAVERSAL COUNTER AT THE EDGE (TRAINS CONTINUE TO s_in)
                    self.H = add_num_trains_continue_graph(self.H, s_out_name, 
                                                                       s_in_name, d_period, NUM_PERIODS)

                    # ADD TO THE END NODE PASSAGE COUNTER
                    self.H = add_passage_count_graph(self.H, s_out_name,
                                                                 d_period, NUM_PERIODS)
                    self.H = add_passage_count_graph(self.H, s_in_name,
                                                                 d_period, NUM_PERIODS)

                    # ADD TO THE STOPPING STATIONS COUNT
                    self.H = add_stop_trains_graph(self.H, s_in_name,
                                                             d_period, NUM_PERIODS)
                else:
                    # ADD TO THE NODE PASSAGE COUNTER
                    d_period = dp[planned_in_dep.hour]
                    self.H = add_passage_count_graph(self.H, s_out_name,
                                                                 d_period, NUM_PERIODS)

                    # ADD TO THE TRAVERSAL COUNTER AT THE EDGE (TRAINS CONTINUE TO s_in)
                    self.H = add_num_trains_continue_graph(self.H, s_out_name, 
                                                                       s_in_name, d_period, NUM_PERIODS)
                    # ADD TO THE PASSAGE TIME
                    delta_time = (planned_in_dep - planned_out_dep).total_seconds()
                    d_period = dp[planned_in_dep.hour]
                    self.H = add_traversal_time_graph(self.H, s_out_name, 
                                                                  s_in_name, delta_time, d_period, NUM_PERIODS)

        if len(track) == 2:
            s1, s2 = track
            s_out_name = s1[0]
            s_in_name = s2[0]

            planned_out_dep = s1[3]
            planned_in_arr = s2[1]

            # ADD TO THE NODE PASSAGE COUNTER (s_out)
            d_period = dp[planned_out_dep.hour]
            self.H = add_passage_count_graph(self.H, s_out_name,
                                                         d_period, NUM_PERIODS)
            # ADD TO THE NODE PASSAGE COUNTER (s_in)
            d_period = dp[planned_in_arr.hour]
            self.H = add_passage_count_graph(self.H, s_in_name,
                                                         d_period, NUM_PERIODS)

            # ADD TO THE STOPPING STATIONS COUNT
            self.H = add_stop_trains_graph(self.H, s_in_name,
                                                     d_period, NUM_PERIODS)

            # ADD TO THE TRAVERSAL COUNTER AT THE EDGE (TRAINS CONTINUE TO s_in)
            self.H = add_num_trains_continue_graph(self.H, s_out_name, 
                                                               s_in_name, d_period, NUM_PERIODS)

            # ADD TO THE PASSAGE TIME
            delta_time = (planned_in_arr - planned_out_dep).total_seconds()
            d_period = dp[planned_in_arr.hour]
            self.H = add_traversal_time_graph(self.H, s_out_name, 
                                                          s_in_name, delta_time, d_period, NUM_PERIODS)
        
        
    H_avg = nx.DiGraph(day = 0, num_periods = NUM_PERIODS)

    for u, data in self.H.nodes(data = True):
        H_avg.add_node(u)
        # STOPPED TRAINS 
        if "stopped_trains" in data:
            avg_stopped_trains = list(np.array(data["stopped_trains"]))
            H_avg.nodes[u]["stopped_trains"] = avg_stopped_trains

        # PASSAGE COUNT
        if "passage_count" in data:
            avg_passage_count = list(np.array(data["passage_count"]))
            H_avg.nodes[u]["passage_count"] = avg_passage_count

    for u, v, data in self.H.edges(data = True):
        H_avg.add_edge(u,v)
        # TRAVERSAL TIME
        if "traversal_time" in data:
            avg_traversal_time = [np.mean(x) if len(x) else 0 for x in data["traversal_time"]]
            H_avg[u][v]["traversal_time"] = avg_traversal_time

        # NUMBER OF TRAINS THAT CONTINUE
        if "num_trains_continue" in data:
            avg_num_trains_continue = list(np.array(data["num_trains_continue"]))
            H_avg[u][v]["num_trains_continue"] = avg_num_trains_continue

    # REMOVE THE EDGES WHICH WILL NOT BE TRAVERSED
    count_no_data = []
    for u, v, data in H_avg.edges(data = True):
        if len(data) == 0:
            count_no_data.append((u,v))
    H_avg.remove_edges_from(count_no_data)

    # REMOVE HANGING NODES FROM THE NETWORK
    nodes_to_remove = []
    for u in H_avg.nodes():
        if H_avg.degree(u) == 0:
            nodes_to_remove.append(u)
    H_avg.remove_nodes_from(nodes_to_remove)

    print(f":: no data on nodes - {len(nodes_to_remove)}")
    return H_avg





# EDGE BASED MODEL

def prepare_network_file_edge_based(self, num_periods, verbose = True):
    NUM_PERIODS = num_periods

    # CHOOSE THE NUMBER OF PERIODS YOU WISH TO SPLIT THE DAY 
    # PERIOD IS THE NUMBER OF HOURS, THUS THERE CAN BE 1,2,4,6,12 OR 24 PERIODS WITH APPRORPIATE 
    dp = get_period_of_day(NUM_PERIODS)
        
    # COUNTER FOR THE DATA FILES CONSIDERED FOR EACH DAY OF THE WEEK
    wday_counter = [0]*7

    # ITERATE OVER THE DAYS 
    count_problems_corrected_track = defaultdict(float)
    for day_counter, fpath in tqdm(enumerate(self.path_list), total = len(self.path_list), disable = not verbose): 
        current_date = fpath[-10:]
        print(f"Working with date :: {current_date}")
        wday_index = datetime.strptime(current_date, "%Y-%m-%d").weekday()
        wday_counter[wday_index] += 1

        train_tracks = upload_schedule(fpath) #load this day's tracks
        counter_successful_tracks = 0.
        for train, track in tqdm(train_tracks.items(), total = len(train_tracks), disable = not verbose): 
            # iterate over the lines of the file, each line is one track
            if any(s in train for s in self.forbidden_trains):
                continue
            # correct the track
            try:
                corrected_track = get_corrected_track(track, self.g_delays)
                counter_successful_tracks += 1
            except:
                count_problems_corrected_track[current_date] += 1
                continue
            # obtain the delays from the track and the times when the train crossed edges
            if len(corrected_track)>2:
                for i, (s1, s2, s3) in enumerate(zip(corrected_track, corrected_track[1:], corrected_track[2:])):
                    s_out_name = s1[0]
                    s_inter_name = s2[0]
                    s_in_name = s3[0]
                    
                    # REMEMBER: we are working with datetime objects
                    planned_out_dep = s1[3]
                    planned_inter_dep = s2[3]
                    planned_in_dep = s3[3]

                    # ADD EDGE TRAVERSAL TIME PER HOUR
                    # NOTE: we add the count to the hour of the crossing of the half (!) of an edge 
                    ### IF NONE VALUE OCCURS HERE => ERROR WITH A TRACK
                    out_inter_delta_time = (planned_inter_dep - planned_out_dep).total_seconds()
                    d_period = dp[planned_inter_dep.hour]
                    self.h_list[wday_index] = add_traversal_time_to_edges(self.h_list[wday_index], s_out_name, s_inter_name,
                                                                     out_inter_delta_time, d_period, NUM_PERIODS)

                    # ADD TO THE PASSAGE COUNTER
                    # NOTE: we add the count to the hour of the crossing of the half (!) of an edge 
                    self.h_list[wday_index] = add_passing_trains_to_edges(self.h_list[wday_index], s_out_name, s_inter_name, 
                                                                 d_period, NUM_PERIODS)
                    
                    # ADD TRAVERSAL BETWEEN EDGES
                    d_period = dp[planned_inter_dep.hour]
                    self.h_list[wday_index] = add_passage_trains_between_edges(self.h_list[wday_index], s_out_name, 
                                                                         s_inter_name, s_in_name, 
                                                                         d_period, NUM_PERIODS)

                    # IF THE IN STATION IS THE ENDING, THEN ADD TRAVERSAL TIME TO THE (INTER-IN) EDGE
                    # AND ADD THE PASSAGE COUNTER TOO
                    if planned_in_dep is None: 
                        planned_in_arr = s3[1]
                        d_period = dp[planned_in_arr.hour]
                        inter_in_delta_time = (planned_in_arr - planned_inter_dep).total_seconds()
                        self.h_list[wday_index] = add_traversal_time_to_edges(self.h_list[wday_index], s_inter_name, s_in_name,
                                                                         inter_in_delta_time, d_period, NUM_PERIODS)
                        # passage counter
                        self.h_list[wday_index] = add_passing_trains_to_edges(self.h_list[wday_index], s_inter_name, s_in_name, 
                                                                 d_period, NUM_PERIODS)
                        # stopped trains (may be here need more development)
                        self.h_list[wday_index] = add_stop_trains_to_edges(self.h_list[wday_index], s_inter_name, s_in_name, 
                                                                 d_period, NUM_PERIODS)
                    
                        
            if len(corrected_track) == 2:
                s1, s2 = corrected_track
                s_out_name = s1[0]
                s_in_name = s2[0]

                planned_out_dep = s1[3]
                planned_in_arr = s2[1]
                d_period = dp[planned_in_arr.hour]
                out_inter_delta_time = (planned_in_arr - planned_out_dep).total_seconds()

                self.h_list[wday_index] = add_traversal_time_to_edges(self.h_list[wday_index], s_out_name, s_in_name,
                                                     out_inter_delta_time, d_period, NUM_PERIODS)

                self.h_list[wday_index] = add_passing_trains_to_edges(self.h_list[wday_index], s_out_name, s_in_name,
                                                                 d_period, NUM_PERIODS)

                self.h_list[wday_index] = add_stop_trains_to_edges(self.h_list[wday_index], s_out_name, s_in_name,
                                                                 d_period, NUM_PERIODS)
        
        count_problems_corrected_track[current_date] = round(count_problems_corrected_track[current_date]/len(train_tracks), 3)
    print(f"Problems during correcting tracks (fraction of total): {count_problems_corrected_track}")

    days_of_the_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    avg_h_list = []
    for day_index, d_week in tqdm(enumerate(days_of_the_week), total = len(days_of_the_week)):

        h = self.h_list[day_index]
        h_avg = nx.DiGraph(day = days_of_the_week[day_index], num_periods = NUM_PERIODS)

        for u, data in h.nodes(data = True):
            h_avg.add_node(u)

            # PASSING TRAINS 
            if "passing_trains" in data:
                avg_passing_trains = list(np.array(data["passing_trains"])/wday_counter[day_index])
                h_avg.nodes[u]["passing_trains"] = avg_passing_trains

            # TRAVERSAL TIME FOR TRAINS
            if "traversal_time" in data:
                avg_traversal_time = [np.mean(x) if len(x) else 0 for x in data["traversal_time"]]
                h_avg.nodes[u]["traversal_time"] = avg_traversal_time

            # STOPPING TRAINS
            if "stopped_trains" in data:
                avg_stopped_trains = list(np.array(data["stopped_trains"])/wday_counter[day_index])
                h_avg.nodes[u]["stopped_trains"] = avg_stopped_trains
        
        for u, v, data in h.edges(data = True):
            h_avg.add_edge(u,v)
            # TRAVERSAL BETWEEN EDGES COUNTS
            if "traversal_btw_edges_count" in data:
                avg_traversal_direction_count = list(np.array(data["traversal_btw_edges_count"])/wday_counter[day_index])
                h_avg[u][v]["traversal_btw_edges_count"] = avg_traversal_direction_count

        # REMOVE THE EDGES WHICH WILL NOT BE TRAVERSED
        count_no_data = []
        for u, v, data in h_avg.edges(data = True):
            if len(data) == 0:
                count_no_data.append((u,v))
        h_avg.remove_edges_from(count_no_data)
        
        # REMOVE HANGING NODES FROM THE NETWORK
        nodes_to_remove = []
        for u in h_avg.nodes():
            if h_avg.degree(u) == 0:
                nodes_to_remove.append(u)
        h_avg.remove_nodes_from(nodes_to_remove)
        
        avg_h_list.append(h_avg)
    return avg_h_list

    
def prepare_network_custom_edge_based(self, schedules_dict, verbose = True):

    NUM_PERIODS = 1
    dp = get_period_of_day(NUM_PERIODS)

    for train, track in tqdm(schedules_dict.items()): 
    # iterate over the lines of the file, each line is one track

        # obtain the delays from the track and the times when the train crossed edges
        if len(track)>2:
            for i, (s1, s2, s3) in enumerate(zip(track, track[1:], track[2:])):
                s_out_name = s1[0]
                s_inter_name = s2[0]
                s_in_name = s3[0]

                # REMEMBER: we are working with datetime objects
                planned_out_dep = s1[3]
                planned_inter_dep = s2[3]
                planned_in_dep = s3[3]

                # ADD EDGE TRAVERSAL TIME PER HOUR
                # NOTE: we add the count to the hour of the crossing of the half (!) of an edge 
                ### IF NONE VALUE OCCURS HERE => ERROR WITH A TRACK
                out_inter_delta_time = (planned_inter_dep - planned_out_dep).total_seconds()
                d_period = dp[planned_inter_dep.hour]
                self.H = add_traversal_time_to_edges(self.H, s_out_name, s_inter_name,
                                                                 out_inter_delta_time, d_period, NUM_PERIODS)

                # ADD TO THE PASSAGE COUNTER
                # NOTE: we add the count to the hour of the crossing of the half (!) of an edge 
                self.H = add_passing_trains_to_edges(self.H, s_out_name, s_inter_name, 
                                                             d_period, NUM_PERIODS)

                # ADD TRAVERSAL BETWEEN EDGES
                d_period = dp[planned_inter_dep.hour]
                self.H = add_passage_trains_between_edges(self.H, s_out_name, 
                                                                     s_inter_name, s_in_name, 
                                                                     d_period, NUM_PERIODS)

                # IF THE IN STATION IS THE ENDING, THEN ADD TRAVERSAL TIME TO THE (INTER-IN) EDGE
                # AND ADD THE PASSAGE COUNTER TOO
                if planned_in_dep is None: 
                    planned_in_arr = s3[1]
                    d_period = dp[planned_in_arr.hour]
                    inter_in_delta_time = (planned_in_arr - planned_inter_dep).total_seconds()
                    self.H = add_traversal_time_to_edges(self.H, s_inter_name, s_in_name,
                                                                     inter_in_delta_time, d_period, NUM_PERIODS)
                    # passage counter
                    self.H = add_passing_trains_to_edges(self.H, s_inter_name, s_in_name, 
                                                             d_period, NUM_PERIODS)
                    # stopped trains (may be here need more development)
                    self.H = add_stop_trains_to_edges(self.H, s_inter_name, s_in_name, 
                                                             d_period, NUM_PERIODS)

        if len(track) == 2:
            s1, s2 = track
            s_out_name = s1[0]
            s_in_name = s2[0]

            planned_out_dep = s1[3]
            planned_in_arr = s2[1]
            d_period = dp[planned_in_arr.hour]
            out_inter_delta_time = (planned_in_arr - planned_out_dep).total_seconds()

            self.H = add_traversal_time_to_edges(self.H, s_out_name, s_in_name,
                                                 out_inter_delta_time, d_period, NUM_PERIODS)

            self.H = add_passing_trains_to_edges(self.H, s_out_name, s_in_name,
                                                             d_period, NUM_PERIODS)

            self.H = add_stop_trains_to_edges(self.H, s_out_name, s_in_name,
                                                             d_period, NUM_PERIODS)


    H_avg = nx.DiGraph(day = 0, num_periods = NUM_PERIODS)

    for u, data in self.H.nodes(data = True):
        H_avg.add_node(u)

        # PASSING TRAINS 
        if "passing_trains" in data:
            avg_passing_trains = list(np.array(data["passing_trains"]))
            H_avg.nodes[u]["passing_trains"] = avg_passing_trains

        # TRAVERSAL TIME FOR TRAINS
        if "traversal_time" in data:
            avg_traversal_time = [np.mean(x) if len(x) else 0 for x in data["traversal_time"]]
            H_avg.nodes[u]["traversal_time"] = avg_traversal_time

        # STOPPING TRAINS
        if "stopped_trains" in data:
            avg_stopped_trains = list(np.array(data["stopped_trains"]))
            H_avg.nodes[u]["stopped_trains"] = avg_stopped_trains

    for u, v, data in self.H.edges(data = True):
        H_avg.add_edge(u,v)
        # TRAVERSAL BETWEEN EDGES COUNTS
        if "traversal_btw_edges_count" in data:
            avg_traversal_direction_count = list(np.array(data["traversal_btw_edges_count"]))
            H_avg[u][v]["traversal_btw_edges_count"] = avg_traversal_direction_count

    # REMOVE THE EDGES WHICH WILL NOT BE TRAVERSED
    count_no_data = []
    for u, v, data in H_avg.edges(data = True):
        if len(data) == 0:
            count_no_data.append((u,v))
    H_avg.remove_edges_from(count_no_data)

    # REMOVE HANGING NODES FROM THE NETWORK
    nodes_to_remove = []
    for u in H_avg.nodes():
        if H_avg.degree(u) == 0:
            nodes_to_remove.append(u)
    H_avg.remove_nodes_from(nodes_to_remove)
    print(count_no_data)
    print(f":: no data on edges - {len(count_no_data)}, edges to remove :: {len(nodes_to_remove)}")
    return H_avg 


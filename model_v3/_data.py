from scipy.signal import savgol_filter
import os
import re
import numpy as np
import json
from collections import defaultdict
from datetime import datetime, timedelta 

from .utils import *

# DEFAULT CONSTANTS

DAYS_OF_THE_WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# COMMON FUNCTIONS

def store_data_from_directory(self, data_dir):
    # STORE LOCATIONS OF DATA FILES
    date_pattern = r"\d{4}-\d{2}-\d{2}"
    files_list = [f for f in sorted(os.listdir(data_dir)) if re.search(date_pattern, f) is not None]
    self.path_list = [data_dir + f for f in files_list]

    self.weekday_dict = defaultdict(list)
    if len(self.path_list):
        for fpath in tqdm(self.path_list, disable = not self.is_global_verbose): 
            wday_index = datetime.strptime(fpath[-10:], "%Y-%m-%d").weekday()
            self.weekday_dict[wday_index].append(fpath)
        if self.is_global_verbose:
            print("""Data is prepared. 
                Access .weekday_dict for the files sorted by days of the week
                Access .path_list for the full list of files""")
    else:
        raise("The data filenames should be in format YEAR-MM-DD, corresponding to the data")

def upload_raw_data(self, data_path):
    # PRELIMINARY STEPS
    self.current_date = data_path[-10:]
    if self.is_global_verbose:
        print(f"\nWorking with file :: {self.current_date}")
    self.start_day_dt = datetime.strptime(self.current_date, "%Y-%m-%d")

    # START UPLOADING THE DATA
    if self.is_global_verbose:
        print("Uploading raw train tracks...")
    self.train_tracks = upload_schedule(data_path) # load this day's tracks
    if self.is_global_verbose:
        print("""SUCCESS! 
            Now you can use the .calculate_model_parameters to prepare model parameters for simulation and then .parse_raw_data to prepare the raw data""")

def upload_custom_train_tracks(self, custom_train_tracks):
    # START UPLOADING THE DATA
    if self.is_global_verbose:
        print("Uploading custom train tracks...")
    self.train_tracks = custom_train_tracks 
    if self.is_global_verbose:
        print("""SUCCESS! 
            Now you can use the .calculate_model_parameters to prepare model parameters for simulation and then .parse_raw_data to prepare the raw data""")

def get_timeseries_from_data(self, num_timestamps, dt, network_for_path_search, node_list, cluster_list = None, 
    is_edge_model = False):
    # CONVERT DATA TO TIME SERIES
    node_ts = np.zeros((num_timestamps, len(node_list)))
    if cluster_list is not None:
        cluster_ts = np.zeros((num_timestamps, len(cluster_list)))

    for train, track in tqdm(self.train_tracks.items(), total = len(self.train_tracks), 
        disable = not self.is_global_verbose,
        desc = "Turning the data into time series of aggregated delays"): 
        # iterate over the lines of the file, each line is one track
        if isinstance(train, str):
            if any(s in train for s in self.forbidden_trains):
                continue
        # correct the track
        try:
            corrected_track = get_corrected_track(track, network_for_path_search)
        except:
            continue
        # obtain the delays from the track and the times when the train crossed edges
        if len(corrected_track) >= 2:
            for (s1, s2) in zip(corrected_track, corrected_track[1:]):
                s_out_name = s1[0]
                s_in_name = s2[0]
                if (
                    (s_in_name not in self.cluster_dict) 
                    or (s_out_name not in self.cluster_dict)
                ):
                    continue
                if is_edge_model:                    
                    index_v = node_list.index((s_out_name, s_in_name))
                    if cluster_list is not None:
                        cluster_s_out = self.cluster_dict[s_out_name]
                        cluster_s_in = self.cluster_dict[s_in_name]
                        index_c = cluster_list.index((cluster_s_out, cluster_s_in))
                else:
                    index_v = node_list.index(s_in_name)
                    if cluster_list is not None:
                        index_c = cluster_list.index(self.cluster_dict[s_in_name])

                # NOTE: we are working with datetime objects
                planned_out_dep = s1[3]
                planned_in_dep = s2[3]
                
                real_out_dep = s1[4]
                real_in_dep = s2[4]
                
                out_dep_delay = s1[-1]
                if not np.isnan(s2[-1]):
                    in_dep_delay = s2[-1]
                else:
                    in_dep_delay = s2[-2]

                if self.start_day_dt is None:
                    self.start_day_dt = datetime(real_out_dep.year, real_out_dep.month, real_out_dep.day)
            
                if real_in_dep is not None:
                    t_delay_account_start = int((planned_out_dep - self.start_day_dt).total_seconds()/dt)
                    t_delay_account_finish = int((planned_in_dep - self.start_day_dt).total_seconds()/dt)
                else:
                    planned_in_arr = s2[1]

                    t_delay_account_start = int((planned_out_dep - self.start_day_dt).total_seconds()/dt)
                    t_delay_account_finish = int((planned_in_arr - self.start_day_dt).total_seconds()/dt)

                if t_delay_account_start < num_timestamps and t_delay_account_finish < num_timestamps:
                    if (t_delay_account_finish - t_delay_account_start) != 0:
                        k = (in_dep_delay - out_dep_delay)/(t_delay_account_finish - t_delay_account_start)
                        
                        for t in range(t_delay_account_start, t_delay_account_finish):
                            node_ts[t][index_v] += out_dep_delay + k*(t - t_delay_account_start)
                            if cluster_list is not None:
                                cluster_ts[t][index_c] += out_dep_delay + k*(t - t_delay_account_start)
                       
                    else:
                        t = t_delay_account_start
                        node_ts[t][index_v] += in_dep_delay
                        if cluster_list is not None:
                            cluster_ts[t][index_c] += out_dep_delay 
    
    if cluster_list is None:
        return node_ts
    else:
        return node_ts, cluster_ts

# NODE BASED MODEL

def parse_raw_data_node_based_model(self, dt, num_timestamps = None):

    # PRELIMINARY STEPS
    if self.current_date is not None:
        print(f"Working with file :: {self.current_date}")
        wday_index = datetime.strptime(self.current_date, "%Y-%m-%d").weekday()
        if self.g.graph["day"] != DAYS_OF_THE_WEEK[wday_index]:
            warnings.warn("The day of the week for the supplied the network doesn't match with the data date. Up to you to stop me.", Warning)

        self.start_day_dt = datetime.strptime(self.current_date, "%Y-%m-%d")
    
    self.real_node_list = list(self.g.nodes())

    if num_timestamps is None:
        num_timestamps = int(1440*60/dt)     # timestamps in the day

    # CONVERT DATA TO TIME SERIES
    
    self.real_node_ts = get_timeseries_from_data(self, num_timestamps, dt, self.g, self.real_node_list)
    
    self.real_node_ts_smooth = savgol_filter(self.real_node_ts, 61, 2, axis = 0)
    if self.is_global_verbose:
        print(f"""Data is stored.
            Access .start_day_dt for the datetime timestamp of the current date
            Access .real_node_ts for the time series of aggregated delays on nodes
            Access .real_node_ts_smooth for their smoothed version
            Access .real_node_list for the ordered list of nodes associated with the time series """)


def parse_raw_data_clustered_node_based_model(self, dt):
 
    wday_index = datetime.strptime(self.current_date, "%Y-%m-%d").weekday()
    if self.g is None:
        raise("No graph parameters were learned. Please use .calculate_model_parameters(g, cluster_dict) first! ")

    if self.g.graph["day"] != DAYS_OF_THE_WEEK[wday_index]:
        warnings.warn("The day of the week for the supplied the network doesn't match with the data date. Up to you to stop me.", Warning)

    self.dt = dt

    # IDENTIFY THE NODE LIST FROM THE NETWORK
    self.node_list = list(self.g.nodes)

    # CONVERT DATA TO TIME SERIES
    num_timestamps = int(1440*60/dt)     # timestamps in the day
    self.node_ts, self.cluster_ts = get_timeseries_from_data(self, num_timestamps, dt, self.g, 
                                    self.node_list, cluster_list = self.cluster_list)
 
    self.cluster_ts_smooth = savgol_filter(self.cluster_ts, 61, 2, axis = 0)
    self.node_ts_smooth = savgol_filter(self.node_ts, 61, 2, axis = 0)
    if self.is_global_verbose:
        print(f"""Data is stored.
            Access .start_day_dt for the datetime timestamp of the current date
            Access .node_ts for the time series of aggregated delays on nodes
            Access .node_ts_smooth for their smoothed version
            Access .node_list for the ordered list of nodes, used to index the time series 

            Access .cluster_ts for the time series of aggregated delays on clusters
            Access .cluster_ts_smooth for their smoothed version
            Access .cluster_list for the ordered list of clusters, used to index the time series""")




# EDGE BASED MODEL

def parse_raw_data_edge_based_model(self, dt, num_timestamps = None):
    # PRELIMINARY STEPS
    if self.current_date is not None:
        print(f"Working with file :: {self.current_date}")
        wday_index = datetime.strptime(self.current_date, "%Y-%m-%d").weekday()
        if self.h.graph["day"] != DAYS_OF_THE_WEEK[wday_index]:
            warnings.warn("The day of the week for the supplied the network doesn't match with the data date. Up to you to stop me.", Warning)

        self.start_day_dt = datetime.strptime(self.current_date, "%Y-%m-%d")
    
    self.real_edge_list = list(self.h.nodes())
    self.dt = dt

    if num_timestamps is None:
        num_timestamps = int(1440*60/dt)     # timestamps in the day

    # CONVERT DATA TO TIME SERIES
    self.real_edge_ts = get_timeseries_from_data(self, num_timestamps, dt, self.g, self.real_edge_list,
                                                 is_edge_model = True)

    self.real_edge_ts_smooth = savgol_filter(self.real_edge_ts, 61, 2, axis = 0)
    if self.is_global_verbose:
        print(f"""Data is stored.
            Access .start_day_dt for the datetime timestamp of the current date
            Access .real_edge_ts for the time series of aggregated delays on edges
            Access .real_edge_ts_smooth for their smoothed version
            Access .real_edge_list for the ordered list of edges associated with the time series """)


def parse_raw_data_clustered_edge_based(self, dt):
    if self.start_day_dt is None:
        raise("You did not upload the data first. Please use .upload_data(data_path) first!")

    wday_index = datetime.strptime(self.current_date, "%Y-%m-%d").weekday()
    if self.h is None:
        raise("No graph parameters were learned. Please use .calculate_model_parameters(g, cluster_dict) first! ")

    if self.h.graph["day"] != DAYS_OF_THE_WEEK[wday_index]:
        warnings.warn("The day of the week for the supplied the network doesn't match with the data date. Up to you to stop me.", Warning)

    self.dt = dt

    # IDENTIFY THE EDGE LIST FROM THE LINK GRAPH
    self.edge_list = list(self.h.nodes)
    self.cluster_edge_list = list(self.h_clustered.nodes)

    # CONVERT DATA TO NUMPY ARRAY OF TIME SERIES
    num_timestamps = int(1440*60/dt)     # timestamps in the day
    self.edge_ts, self.edge_cluster_ts = get_timeseries_from_data(self, num_timestamps, dt, self.g, 
                                            self.edge_list, cluster_list = self.cluster_edge_list,
                                            is_edge_model = True)
    
    self.edge_cluster_ts_smooth = savgol_filter(self.edge_cluster_ts, 61, 2, axis = 0)
    self.edge_ts_smooth = savgol_filter(self.edge_ts, 61, 2, axis = 0)
    if self.is_global_verbose:
        print(f"""Data is stored.
            Access .start_day_dt for the datetime timestamp of the current date
            Access .edge_ts for the time series of aggregated delays on edges
            Access .edge_ts_smooth for their smoothed version
            Access .edge_list for the ordered list of nodes, used to index the time series 

            Access .edge_cluster_ts for the time series of aggregated delays on clusters
            Access .edge_cluster_ts_smooth for their smoothed version
            Access .cluster_edge_list for the ordered list of clusters, used to index the time series""")




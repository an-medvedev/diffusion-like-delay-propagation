import numpy as np
from datetime import datetime, timedelta

### ALL PURPOSE MODEL

def simulate_model(g, G, B_vector, start_simulation_dt, T_duration, dt, initial_delay_vec, verbose = False):
    # NOTE: if it blows up - use lower resolution
    # initial_delay_vec must contain the vector of initial delays
    # ordering of nodes must be the same as in h.nodes()

    # PRELIMINARY SETUP
    steps = int(T_duration/dt)
    N = len(g)  # number of nodes in the railway network
    
    # the matrix DD contains in column i the complete timeseries of delays at edge i
    # INITIALIZING THE DELAY MARTIX
    delay_ts_matrix = np.zeros((steps, N))
    
    # INITIAL CONDITIONS
    delay_ts_matrix[0,:] = initial_delay_vec

    # START OF SIMULATION
    # Do the timestepping, which is basically the matrix multiplication
    B_diag = np.diag(B_vector)
    for i in range(0, steps-1):
        D = delay_ts_matrix[i, :]
        if verbose:
            print(f"Step : {i} - sum of all delays in the system: {np.sum(D)}")
        Dnew = D + dt*(np.dot((G - B_diag), D))
        delay_ts_matrix[i+1,:]= Dnew
        
    # TIMESTAMPS OF THE MODELING 
    tt = np.linspace(0, T_duration - dt, steps)
    date_range = [start_simulation_dt + timedelta(seconds = int(x)) for x in tt]
    return tt, date_range, delay_ts_matrix

# ### EDGE BASED MODEL

# def simulate_edge_based_model(h, G, B_vector, start_simulation_dt, T_duration, dt, initial_delay_vec, verbose = False):
#     # NOTE: if it blows up - use lower resolution
#     # initial_delay_vec must contain the vector of initial delays
#     # ordering of nodes must be the same as in h.nodes()

#     # PRELIMINARY SETUP
#     steps = int(T_duration/dt)
#     M = len(h)  # number of edges in the railway network
    
#     # the matrix DD contains in column i the complete timeseries of delays at edge i
#     # INITIALIZING THE DELAY MARTIX
#     delay_ts_matrix = np.zeros((steps, M))
    
#     # INITIAL CONDITIONS
#     delay_ts_matrix[0,:] = initial_delay_vec

#     # START OF SIMULATION
#     # Do the timestepping, which is basically the matrix multiplication
#     B_diag = np.diag(B_vector)
#     for i in range(0, steps-1):
#         D = delay_ts_matrix[i, :]
#         if verbose:
#             print(f"Step : {i} - sum of all delays in the system: {np.sum(D)}")
#         Dnew = D + dt*(np.dot(G - B_diag, D))
#         delay_ts_matrix[i+1,:] = Dnew.T
        
#     # TIMESTAMPS OF THE MODELING 
#     tt = np.linspace(0, T_duration - dt, steps)
#     date_range = [start_simulation_dt + timedelta(seconds = int(x)) for x in tt]
#     return tt, date_range, delay_ts_matrix

    
# def simulate_clustered_edge_based_model(h_clustered, G, B_vector, start_simulation_dt, T_duration, initial_delay_vec, verbose = False):
#     # NOTE: if it blows up - use lower resolution
#     # initial_delay_vec must contain the vector of initial delays
#     # ordering of nodes must be the same as in h.nodes()

#     # PRELIMINARY SETUP
#     steps = int(T_duration/dt)
#     M = len(h_clustered)  # number of nodes in the edge network
    
#     # the matrix DD contains in column i the complete timeseries of delays at edge i
#     # INITIALIZING THE DELAY MARTIX
#     delay_ts_matrix = np.zeros((self.steps, M))
    
#     # INITIAL CONDITIONS
#     delay_ts_matrix[0,:] = initial_delay_vec

#     # START OF SIMULATION
#     # Do the timestepping, which is basically the matrix multiplication
#     B_diag = np.diag(B_vector)
#     for i in range(0, steps-1):
#         D = delay_ts_matrix[i, :]
#         if verbose:
#             print(f"Step : {i} - sum of all delays in the system: {np.sum(D)}")
#         Dnew = D + dt*(np.dot(G - B_diag, D))
#         delay_ts_matrix[i+1,:]= Dnew
        
#     # TIMESTAMPS OF THE MODELING 
#     tt = np.linspace(0, T_duration - dt, steps)
#     date_range = [start_simulation_dt + timedelta(seconds = int(x)) for x in tt]
#     return tt, date_range, delay_ts_matrix

    

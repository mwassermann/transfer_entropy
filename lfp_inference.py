import os
import numpy as np
import pandas as pd
from allensdk.brain_observatory.ecephys.ecephys_session import EcephysSession
from idtxl.data import Data
from idtxl.multivariate_te import MultivariateTE
from idtxl.visualise_graph import plot_network
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpi4py import MPI

# ---- USER PARAMETERS ----
SESSION_NWB_PATH = "session_715093703.nwb"  # Path to your session NWB file
LFP_NWB_DIR = "."  # Directory containing probe_xxx_lfp.nwb files
ROIS = ["AM", "PM", "AL", "RL", "LM", "V1"]
N_CHANNELS_PER_ROI = 3
N_REPLICATIONS = 1  # For LFP, usually 1
NORMALISE = True    # Z-score per channel

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# 1. All ranks load session metadata
session = EcephysSession.from_nwb_path(SESSION_NWB_PATH)
channels = session.channels

# 2. Split ROIs among ranks
my_rois = [roi for i, roi in enumerate(ROIS) if i % size == rank]

my_selected = []
for roi in my_rois:
    roi_channels = channels[channels['ecephys_structure_acronym'] == roi]
    for probe_id in roi_channels['probe_id'].unique():
        probe_channels = roi_channels[roi_channels['probe_id'] == probe_id]
        lfp = session.get_lfp(probe_id)
        local_indices = probe_channels['local_index'].values
        lfp_data = lfp.data[:, local_indices]
        variances = np.var(lfp_data, axis=0)
        top_idx = np.argsort(variances)[-N_CHANNELS_PER_ROI:]
        for idx in top_idx:
            my_selected.append({
                'roi': roi,
                'probe_id': probe_id,
                'local_index': local_indices[idx],
                'variance': variances[idx],
                'trace': lfp_data[:, idx]
            })

# 3. Gather all selected channels/traces to rank 0
all_selected = comm.gather(my_selected, root=0)

if rank == 0:
    # Flatten and assemble data
    selected_channels = [item for sublist in all_selected for item in sublist]
    data_arr = np.stack([ch['trace'] for ch in selected_channels])
    data_arr = data_arr[:, :, np.newaxis]  # (nodes, samples, replications)

    print(f"Final data shape for IDTxl: {data_arr.shape} (nodes, samples, replications)")

    # ---- RUN TE INFERENCE (OPTIONAL) ----
    # You can now use your IDTxl pipeline as in te_inference.py
    n_nodes = data_arr.shape[0]
    max_lag = 5  # or as desired
    network_analysis = MultivariateTE()
    settings = {
        "cmi_estimator": "JidtGaussianCMI",
        "max_lag_sources": max_lag,
        "min_lag_sources": 1,
        "MPI": False,
        "max_workers": 0,
    }
    data = Data(data_arr, dim_order='psr')
    results = network_analysis.analyse_network(settings=settings, data=data)

    # ---- PLOT NETWORK ----
    results.print_edge_list(weights="max_te_lag", fdr=False)
    plot_network(results=results, weights="max_te_lag", fdr=False)
    plt.savefig('results/network_lfp.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ---- SAVE SELECTED CHANNELS ----
    selected_df = pd.DataFrame(selected_channels)
    selected_df.to_csv("results/selected_lfp_channels.csv", index=False)
    print("Selected channels saved to results/selected_lfp_channels.csv")
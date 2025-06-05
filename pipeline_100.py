import numpy as np
import matplotlib.pyplot as plt
from typing import Literal, Optional, Tuple, Dict, List
import seaborn as sns
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing
import pickle
from pathlib import Path
import argparse
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

from idtxl.data import Data
from idtxl.multivariate_te import MultivariateTE
from idtxl.visualise_graph import plot_network
from synthetic_data_generation import (
    generate_erdos_renyi_network,
    simulate_var,
    simulate_clm
)

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def generate_synthetic_data(
    n_nodes: int = 5,
    n_samples: int = 1000,
    n_replications: int = 5,
    model_type: Literal["VAR", "CLM"] = "VAR",
    avg_degree: float = 2,
    max_lag: int = 5,
    beta: float = 0.5,
    alpha_sum: float = 0.4,
    noise_std: float = 0.1,
    r_logistic: float = 4.0,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic time series data with known network structure and delays.
    
    Args:
        n_nodes: Number of nodes in the network
        n_samples: Number of time samples to generate
        n_replications: Number of replications of the time series
        model_type: Type of model to use ("VAR" or "CLM")
        avg_degree: Average degree of the network
        max_lag: Maximum time lag in the network
        beta: Coupling strength
        alpha_sum: Sum of autoregressive coefficients
        noise_std: Standard deviation of noise
        r_logistic: Logistic map parameter (only used for CLM)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple containing:
        - Generated time series data (shape: n_processes x n_samples x n_replications)
        - True adjacency matrix
        - True connection delays
    """
    try:
        print(f"\nGenerating {model_type} data with parameters:")
        print(f"n_nodes={n_nodes}, n_samples={n_samples}, n_replications={n_replications}")
        print(f"avg_degree={avg_degree}, max_lag={max_lag}, beta={beta}")
        print(f"alpha_sum={alpha_sum}, noise_std={noise_std}")
        if model_type == "CLM":
            print(f"r_logistic={r_logistic}")
        
        # Generate random network structure
        adj_matrix, connection_lags = generate_erdos_renyi_network(
            N=n_nodes,
            avg_degree=avg_degree,
            max_lag=max_lag,
            seed=seed
        )
        print(f"Generated network with {np.sum(adj_matrix)} connections")
        
        # Generate time series data for each replication
        data = np.zeros((n_nodes, n_samples, n_replications))
        for rep in range(n_replications):
            print(f"\nGenerating replication {rep + 1}/{n_replications}")
            try:
                if model_type == "VAR":
                    # Transpose the data to get shape (n_nodes, n_samples)
                    data[:, :, rep] = simulate_var(
                        adj_matrix=adj_matrix,
                        connection_lags=connection_lags,
                        N=n_nodes,
                        T=n_samples,
                        beta=beta,
                        alpha_sum=alpha_sum,
                        noise_std=noise_std,
                        max_lag_val=max_lag,
                        seed=seed + rep if seed is not None else None
                    ).T  # Transpose here
                else:  # CLM
                    # Transpose the data to get shape (n_nodes, n_samples)
                    data[:, :, rep] = simulate_clm(
                        adj_matrix=adj_matrix,
                        connection_lags=connection_lags,
                        N=n_nodes,
                        T=n_samples,
                        beta=beta,
                        alpha_sum=alpha_sum,
                        noise_std=noise_std,
                        r_logistic=r_logistic,
                        max_lag_val=max_lag,
                        seed=seed + rep if seed is not None else None
                    ).T  # Transpose here
                print(f"Successfully generated data for replication {rep + 1}")
            except Exception as e:
                print(f"Error generating data for replication {rep + 1}:")
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {str(e)}")
                raise
        
        print("\nData generation completed successfully")
        print(f"Final data shape: {data.shape}")
        return data, adj_matrix, connection_lags
        
    except Exception as e:
        print(f"\nError in generate_synthetic_data:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        raise

def analyze_target_parallel(target, data_obj, settings):
    """Helper function to analyze a single target node in parallel"""
    try:
        # Create a new analysis object for each target
        target_analysis = MultivariateTE()
        
        # Analyze single target
        target_results = target_analysis.analyse_single_target(
            settings=settings,
            data=data_obj,
            target=target
        )
        
        # Get delays and sources based on FDR setting
        fdr = settings.get('fdr_correction', False)
        try:
            delays = target_results.get_target_delays(target=target, criterion="max_te", fdr=fdr)
            sources = target_results.get_target_sources(target=target, fdr=fdr)
        except (RuntimeError, KeyError) as e:
            if fdr:
                # If FDR fails, try without FDR
                print(f"FDR analysis failed for target {target}, trying without FDR...")
                delays = target_results.get_target_delays(target=target, criterion="max_te", fdr=False)
                sources = target_results.get_target_sources(target=target, fdr=False)
            else:
                raise e
        
        # Store results before cleanup
        result = dict(zip(sources, delays))
        
        # Clean up intermediate results
        del target_results
        del target_analysis
        
        return target, result
    except (RuntimeError, KeyError) as e:
        print(f"Error analyzing target {target}: {str(e)}")
        return target, {}

def analyze_network_parallel(
    data: np.ndarray,
    max_lag: int = 5,
    min_lag: int = 1,
    cmi_estimator: str = "PythonKraskovCMI",
    n_perm_max_stat: int = 200,
    n_perm_min_stat: int = 200,
    n_perm_omnibus: int = 500,
    n_perm_max_seq: int = 500,
    alpha_max_stat: float = 0.05,
    alpha_min_stat: float = 0.05,
    alpha_omnibus: float = 0.05,
    alpha_max_seq: float = 0.05,
    fdr_correction: bool = True,
    n_jobs: int = 4
) -> Dict[int, Dict[int, int]]:
    """
    Analyze network using multivariate transfer entropy with parallel target analysis.
    """
    # Define settings
    settings = {
        "cmi_estimator": cmi_estimator,
        "max_lag_sources": max_lag,
        "min_lag_sources": min_lag,
        "n_perm_max_stat": n_perm_max_stat,
        "n_perm_min_stat": n_perm_min_stat,
        "n_perm_omnibus": n_perm_omnibus,
        "n_perm_max_seq": n_perm_max_seq,
        "alpha_max_stat": alpha_max_stat,
        "alpha_min_stat": alpha_min_stat,
        "alpha_omnibus": alpha_omnibus,
        "alpha_max_seq": alpha_max_seq,
        "fdr_correction": fdr_correction
    }
    
    # Create Data object
    data_obj = Data(data, dim_order='psr')
    
    # Process targets in smaller batches to reduce memory usage
    inferred_delays = {}
    batch_size = 1  # Process one target at a time
    
    # Calculate optimal number of parallel jobs based on available targets and MPI processes
    n_targets = data.shape[0]
    actual_n_jobs = min(n_jobs, n_targets)  # Don't create more jobs than targets
    
    # Use MPIPoolExecutor for distributed computing
    with MPIPoolExecutor(max_workers=actual_n_jobs) as executor:
        for batch_start in range(0, n_targets, batch_size):
            batch_end = min(batch_start + batch_size, n_targets)
            batch_targets = range(batch_start, batch_end)
            
            print(f"\nProcessing targets {batch_start} to {batch_end-1}...")
            
            futures = []
            for target in batch_targets:
                futures.append(executor.submit(analyze_target_parallel, target, data_obj, settings))
            
            for future in as_completed(futures):
                try:
                    target, delays = future.result()
                    inferred_delays[target] = delays
                except Exception as e:
                    print(f"Error processing target: {str(e)}")
                    inferred_delays[target] = {}
            
            # Force garbage collection after each batch
            import gc
            gc.collect()
    
    return inferred_delays

def analyze_network(
    data: np.ndarray,
    max_lag: int = 5,
    min_lag: int = 1,
    cmi_estimator: str = "PythonKraskovCMI",
    n_perm_max_stat: int = 200,
    n_perm_min_stat: int = 200,
    n_perm_omnibus: int = 500,
    n_perm_max_seq: int = 500,
    alpha_max_stat: float = 0.05,
    alpha_min_stat: float = 0.05,
    alpha_omnibus: float = 0.05,
    alpha_max_seq: float = 0.05,
    fdr_correction: bool = True
) -> Dict[int, Dict[int, int]]:
    """
    Analyze network using multivariate transfer entropy.
    
    Args:
        data: Time series data (shape: n_processes x n_samples x n_replications)
        max_lag: Maximum lag to consider
        min_lag: Minimum lag to consider
        cmi_estimator: CMI estimator to use
        n_perm_*: Number of permutations for various statistical tests
        alpha_*: Significance levels for various statistical tests
        fdr_correction: Whether to apply FDR correction
        
    Returns:
        Dictionary mapping targets to their inferred source delays
    """
    # Initialize analysis object
    network_analysis = MultivariateTE()
    
    # Define settings
    settings = {
        "cmi_estimator": cmi_estimator,
        "max_lag_sources": max_lag,
        "min_lag_sources": min_lag,
        "n_perm_max_stat": n_perm_max_stat,
        "n_perm_min_stat": n_perm_min_stat,
        "n_perm_omnibus": n_perm_omnibus,
        "n_perm_max_seq": n_perm_max_seq,
        "alpha_max_stat": alpha_max_stat,
        "alpha_min_stat": alpha_min_stat,
        "alpha_omnibus": alpha_omnibus,
        "alpha_max_seq": alpha_max_seq,
        "fdr_correction": fdr_correction
    }
    
    # Create Data object
    data_obj = Data(data, dim_order='psr')  # Specify dimension order: processes, samples, replications
    
    # Run analysis
    results = network_analysis.analyse_network(settings=settings, data=data_obj)
    
    # Parallelize target analysis
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = []
        for target in range(data.shape[0]):
            futures.append(executor.submit(analyze_target, results, target))
        
        inferred_delays = {}
        for future in as_completed(futures):
            target, delays = future.result()
            inferred_delays[target] = delays
    
    return inferred_delays

def evaluate_delay_reconstruction(
    true_delays: Dict[int, List[int]],
    inferred_delays: Dict[int, List[int]],
    adj_matrix: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate how well the delays were reconstructed.
    
    Args:
        true_delays: Dictionary of true delays
        inferred_delays: Dictionary of inferred delays
        adj_matrix: True adjacency matrix
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Initialize metrics
    total_edges = np.sum(adj_matrix)
    correct_edges = 0
    correct_delays = 0
    delay_errors = []
    
    # Compare true and inferred networks
    for target in range(adj_matrix.shape[0]):
        for source in range(adj_matrix.shape[1]):
            if adj_matrix[source, target]:  # True edge exists
                # Check if edge was detected
                if target in inferred_delays and source in inferred_delays[target]:
                    correct_edges += 1
                    # Check if delay was correctly inferred
                    true_delay = true_delays[source, target]
                    inferred_delay = inferred_delays[target][source]
                    if true_delay == inferred_delay:
                        correct_delays += 1
                    delay_errors.append(abs(true_delay - inferred_delay))
    
    # Calculate metrics
    edge_detection_rate = correct_edges / total_edges if total_edges > 0 else 0
    delay_accuracy = correct_delays / total_edges if total_edges > 0 else 0
    mean_delay_error = np.mean(delay_errors) if delay_errors else float('inf')
    
    return {
        "edge_detection_rate": edge_detection_rate,
        "delay_accuracy": delay_accuracy,
        "mean_delay_error": mean_delay_error
    }

def visualize_time_series(data: np.ndarray, save_path: str):
    """
    Visualize the generated time series data.
    
    Args:
        data: Time series data (shape: n_processes x n_samples x n_replications)
        save_path: Path to save the visualization
    """
    n_nodes = data.shape[0]
    n_samples = data.shape[1]
    n_replications = data.shape[2]
    
    # Create figure with subplots for each node
    fig, axes = plt.subplots(n_nodes, 1, figsize=(15, 3*n_nodes))
    if n_nodes == 1:
        axes = [axes]
    
    # Plot each node's time series
    for node in range(n_nodes):
        ax = axes[node]
        # Plot all replications
        for rep in range(n_replications):
            ax.plot(data[node, :, rep], alpha=0.5, label=f'Rep {rep+1}')
        ax.set_title(f'Node {node+1} Time Series')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        if node == 0:
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_network(adj_matrix: np.ndarray, connection_lags: np.ndarray, save_path: str):
    """
    Visualize the generated network structure.
    
    Args:
        adj_matrix: Adjacency matrix
        connection_lags: Matrix of connection delays (can be numpy array, dict with tuple keys, or int)
        save_path: Path to save the visualization
    """
    try:
        print(f"\nDebug - connection_lags type: {type(connection_lags)}")
        print(f"Debug - connection_lags value: {connection_lags}")
        
        n_nodes = adj_matrix.shape[0]
        
        # Create figure with larger size for more nodes
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot adjacency matrix with adjusted font size
        sns.heatmap(adj_matrix, ax=ax1, cmap='YlOrRd', vmin=0, vmax=1)
        ax1.set_title('Adjacency Matrix', fontsize=12)
        ax1.set_xlabel('Source Node', fontsize=10)
        ax1.set_ylabel('Target Node', fontsize=10)
        
        # Adjust tick labels for better readability
        ax1.set_xticks(np.arange(n_nodes) + 0.5)
        ax1.set_yticks(np.arange(n_nodes) + 0.5)
        ax1.set_xticklabels(range(n_nodes), fontsize=8)
        ax1.set_yticklabels(range(n_nodes), fontsize=8)
        
        # Handle different types of connection_lags
        if isinstance(connection_lags, dict):
            print("Debug - Converting dictionary to matrix")
            # Create a matrix of zeros
            lags_matrix = np.zeros((n_nodes, n_nodes))
            # Fill in the values from the dictionary
            for (source, target), delay in connection_lags.items():
                lags_matrix[source, target] = delay
            connection_lags = lags_matrix
        elif isinstance(connection_lags, (int, float)):
            print("Debug - Converting single number to matrix")
            # If it's a single number, create a matrix with that value
            connection_lags = np.full((n_nodes, n_nodes), float(connection_lags))
        elif isinstance(connection_lags, np.ndarray):
            print("Debug - Using numpy array directly")
            # Ensure it's 2D
            if connection_lags.ndim == 1:
                connection_lags = connection_lags.reshape(n_nodes, n_nodes)
        else:
            print("Debug - Creating zero matrix for unknown type")
            # If it's neither dict nor numpy array, create zeros
            connection_lags = np.zeros((n_nodes, n_nodes))
        
        print(f"Debug - Final connection_lags shape: {connection_lags.shape}")
        print(f"Debug - Final connection_lags type: {type(connection_lags)}")
        
        # Plot connection delays with adjusted font size
        max_delay = np.max(connection_lags) if connection_lags.size > 0 else 1
        sns.heatmap(connection_lags, ax=ax2, cmap='viridis', vmin=0, vmax=max_delay)
        ax2.set_title('Connection Delays', fontsize=12)
        ax2.set_xlabel('Source Node', fontsize=10)
        ax2.set_ylabel('Target Node', fontsize=10)
        
        # Adjust tick labels for better readability
        ax2.set_xticks(np.arange(n_nodes) + 0.5)
        ax2.set_yticks(np.arange(n_nodes) + 0.5)
        ax2.set_xticklabels(range(n_nodes), fontsize=8)
        ax2.set_yticklabels(range(n_nodes), fontsize=8)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error in visualize_network:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("Skipping network visualization...")
        plt.close('all')  # Make sure to close any open figures

def save_results(results: Dict, save_dir: str, trial_idx: int):
    """
    Save analysis results and metadata.
    
    Args:
        results: Dictionary containing results and metadata
        save_dir: Directory to save results
        trial_idx: Trial index
    """
    # Create trial directory
    trial_dir = Path(save_dir) / f"trial_{trial_idx}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    with open(trial_dir / "results.pkl", "wb") as f:
        pickle.dump(results, f)
    
    # Save metadata as text
    with open(trial_dir / "metadata.txt", "w") as f:
        for key, value in results["metadata"].items():
            f.write(f"{key}: {value}\n")

def run_single_trial(
    trial_idx: int,
    n_nodes: int,
    n_samples: int,
    n_replications: int,
    model_type: str,
    max_lag: int,
    seed: Optional[int] = None,
    save_dir: str = "results"
) -> Dict[str, float]:
    """Run a single trial of the analysis pipeline."""
    try:
        print(f"\n{'='*50}")
        print(f"Trial {trial_idx + 1}")
        print(f"{'='*50}")
        
        # Generate data
        print("\nGenerating synthetic data...")
        data, adj_matrix, connection_lags = generate_synthetic_data(
            n_nodes=n_nodes,
            n_samples=n_samples,
            n_replications=n_replications,
            model_type=model_type,
            max_lag=max_lag,
            seed=seed + trial_idx if seed is not None else None
        )
        print(f"Generated network with {np.sum(adj_matrix)} connections")
        print(f"Connection lags type: {type(connection_lags)}")
        print(f"Connection lags shape: {connection_lags.shape if hasattr(connection_lags, 'shape') else 'N/A'}")
        
        # Create results directory
        trial_dir = Path(save_dir) / f"trial_{trial_idx}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        
        # Visualize generated data
        print("\nVisualizing generated data...")
        visualize_time_series(data, trial_dir / "time_series.png")
        visualize_network(adj_matrix, connection_lags, trial_dir / "network.png")
        
        # Analyze network
        print("\nAnalyzing network...")
        inferred_delays = analyze_network_parallel(
            data=data,
            max_lag=max_lag
        )
        
        # Evaluate results
        print("\nEvaluating results...")
        trial_metrics = evaluate_delay_reconstruction(
            true_delays=connection_lags,
            inferred_delays=inferred_delays,
            adj_matrix=adj_matrix
        )
        
        # Calculate metadata values safely
        try:
            if isinstance(connection_lags, np.ndarray):
                max_delay = int(np.max(connection_lags))
                min_delay = int(np.min(connection_lags[connection_lags > 0]))
            elif isinstance(connection_lags, dict):
                max_delay = max(connection_lags.values())
                min_delay = min(connection_lags.values())
            else:
                max_delay = int(connection_lags)
                min_delay = int(connection_lags)
        except Exception as e:
            print(f"Warning: Error calculating delays: {str(e)}")
            max_delay = 0
            min_delay = 0
        
        # Save results and metadata
        results = {
            "data": data,
            "adj_matrix": adj_matrix,
            "connection_lags": connection_lags,
            "inferred_delays": inferred_delays,
            "metrics": trial_metrics,
            "metadata": {
                "trial_idx": trial_idx,
                "n_nodes": n_nodes,
                "n_samples": n_samples,
                "n_replications": n_replications,
                "model_type": model_type,
                "max_lag": max_lag,
                "seed": seed + trial_idx if seed is not None else None,
                "total_connections": int(np.sum(adj_matrix)),
                "avg_degree": float(np.mean(np.sum(adj_matrix, axis=0))),
                "max_delay": max_delay,
                "min_delay": min_delay,
            }
        }
        
        save_results(results, save_dir, trial_idx)
        
        # Print trial summary
        print("\nTrial Summary:")
        print(f"Edge Detection Rate: {trial_metrics['edge_detection_rate']:.3f}")
        print(f"Delay Accuracy: {trial_metrics['delay_accuracy']:.3f}")
        print(f"Mean Delay Error: {trial_metrics['mean_delay_error']:.3f}")
        
        return trial_metrics
        
    except Exception as e:
        print(f"\nError in trial {trial_idx}:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        return {
            "edge_detection_rate": 0.0,
            "delay_accuracy": 0.0,
            "mean_delay_error": float('inf')
        }

def run_delay_analysis_pipeline(
    n_nodes: int = 5,
    n_samples: int = 1000,
    n_replications: int = 10,
    model_type: Literal["VAR", "CLM"] = "VAR",
    max_lag: int = 5,
    n_trials: int = 5,
    seed: Optional[int] = None,
    n_jobs: int = 4,
    save_dir: str = "results"
) -> Dict[str, List[float]]:
    """
    Run complete pipeline for analyzing delay reconstruction performance.
    """
    metrics = {
        "edge_detection_rate": [],
        "delay_accuracy": [],
        "mean_delay_error": []
    }
    
    print(f"\nStarting delay analysis pipeline with {model_type} model")
    print(f"Parameters: n_nodes={n_nodes}, n_samples={n_samples}, n_replications={n_replications}, max_lag={max_lag}")
    print(f"Running {n_trials} trials...\n")
    
    # Process trials sequentially instead of in parallel
    for trial_idx in range(n_trials):
        try:
            print(f"\nProcessing trial {trial_idx + 1}/{n_trials}")
            trial_metrics = run_single_trial(
                trial_idx=trial_idx,
                n_nodes=n_nodes,
                n_samples=n_samples,
                n_replications=n_replications,
                model_type=model_type,
                max_lag=max_lag,
                seed=seed + trial_idx if seed is not None else None,
                save_dir=save_dir
            )
            
            # Store metrics
            for metric, value in trial_metrics.items():
                metrics[metric].append(value)
                
        except Exception as e:
            print(f"\nError in trial {trial_idx}:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            import traceback
            print("\nFull traceback:")
            traceback.print_exc()
            continue
    
    # Print overall summary
    print(f"\n{'='*50}")
    print("Overall Summary Statistics:")
    print(f"{'='*50}")
    for metric, values in metrics.items():
        if values:  # Only print if we have values
            print(f"\n{metric}:")
            print(f"  Mean: {np.mean(values):.3f}")
            print(f"  Std:  {np.std(values):.3f}")
            print(f"  Min:  {np.min(values):.3f}")
            print(f"  Max:  {np.max(values):.3f}")
    
    return metrics

def plot_results(metrics: Dict[str, List[float]], model_type: str, save_dir: str = "results"):
    """
    Plot evaluation metrics and save them to files.
    
    Args:
        metrics: Dictionary containing evaluation metrics
        model_type: Type of model used ("VAR" or "CLM")
        save_dir: Directory to save plots (default: "results")
    """
    # Create results directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot edge detection rate
    sns.histplot(metrics["edge_detection_rate"], ax=axes[0])
    axes[0].set_title("Edge Detection Rate")
    axes[0].set_xlabel("Rate")
    
    # Plot delay accuracy
    sns.histplot(metrics["delay_accuracy"], ax=axes[1])
    axes[1].set_title("Delay Accuracy")
    axes[1].set_xlabel("Accuracy")
    
    # Plot mean delay error
    sns.histplot(metrics["mean_delay_error"], ax=axes[2])
    axes[2].set_title("Mean Delay Error")
    axes[2].set_xlabel("Error")
    
    plt.suptitle(f"Delay Reconstruction Performance - {model_type} Model")
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(save_dir, f"delay_reconstruction_{model_type.lower()}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {save_path}")
    
    # Close figure to free memory
    plt.close(fig)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run transfer entropy analysis pipeline')
    parser.add_argument('--n_nodes', type=int, default=15, help='Number of nodes in the network')
    parser.add_argument('--n_samples', type=int, default=1000, help='Number of time samples')
    parser.add_argument('--n_replications', type=int, default=10, help='Number of replications')
    parser.add_argument('--model_type', type=str, default='VAR', choices=['VAR', 'CLM'], help='Type of model to use')
    parser.add_argument('--max_lag', type=int, default=5, help='Maximum time lag')
    parser.add_argument('--n_trials', type=int, default=5, help='Number of trials to run')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--n_jobs', type=int, default=4, help='Number of parallel jobs')
    parser.add_argument('--save_dir', type=str, default='results', help='Directory to save results')
    
    args = parser.parse_args()
    
    # Only run analysis on rank 0 to avoid duplicate work
    if rank == 0:
        try:
            print(f"\n{'='*80}")
            print(f"Starting analysis for {args.model_type} model")
            print(f"Running on {size} MPI processes")
            print(f"{'='*80}")
            
            metrics = run_delay_analysis_pipeline(
                n_nodes=args.n_nodes,
                n_samples=args.n_samples,
                n_replications=args.n_replications,
                model_type=args.model_type,
                max_lag=args.max_lag,
                n_trials=args.n_trials,
                seed=args.seed,
                n_jobs=args.n_jobs,
                save_dir=args.save_dir
            )
            
            # Print summary statistics
            print(f"\nSummary Statistics for {args.model_type}:")
            for metric, values in metrics.items():
                if values:  # Only print if we have values
                    print(f"{metric}:")
                    print(f"  Mean: {np.mean(values):.3f}")
                    print(f"  Std:  {np.std(values):.3f}")
            
            # Plot results and save to files
            plot_results(metrics, args.model_type, save_dir=args.save_dir)
            
        except Exception as e:
            print(f"\nError occurred during {args.model_type} analysis:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            import traceback
            print("\nFull traceback:")
            traceback.print_exc()
            print("\nAnalysis failed...")
            exit(1)
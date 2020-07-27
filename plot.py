#######
#
# Script to reproduce learning curve plots 
#
# Usage:
#
#   1) python plot.py
#       
#   This will produce a plot of training loss over iterations, averaged over all
#   folds in a single experiment
#######

import glob
from typing import List

import fire
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot(
    fig_title: str = 'dim_comparison',
    ewm_epoch_span: int = 25,
):

    if fig_title == 'dim_comparison':
        experiment_dirs: List[str] = [
            'data/logs/sampler_edim05',
            'data/logs/sampler_edim1',
            'data/logs/sampler_edim2',
            'data/logs/sampler_edim4',
            'data/logs/sampler_edim8',
        ]
        experiment_names = [
            'Dim. Ratio: 0.5',
            'Dim. Ratio: 1',
            'Dim. Ratio: 2',
            'Dim. Ratio: 4',
            'Dim. Ratio: 8',
        ]
        col = 'experiment'
    elif fig_title == 'baseline':
        experiment_dirs: List[str] = ['data/logs/sum_cosine_distance']
        experiment_names = ['Embedding Dim. Ratio 2']
        col = None
    elif fig_title == 'small':
        experiment_dirs: List[str] = ['data/logs/sum_cosine_distance_small']
        experiment_names = ['Small Training Sequences']
        col = None
    elif fig_title == 'horizon':
        experiment_dirs: List[str] = ['data/logs/sum_cosine_distance_horizon']
        experiment_names = ['Long Forecast Horizon']
        col = None
    else:
        raise ValueError("experiment must be one of 'dim_comparison', 'baseline', 'small', or 'horizon'")

    prefix_runs = ['run-DeepAR-Cat-Var', 'run-DeepAR-Embed-Agg-Cosine', 'run-DeepAR-Embed-Agg-L2']
    method_names = ['DeepAR', '+ Cosine Distance', '+ Squared L2']
    
    averaged_runs = pd.DataFrame([])
    for experiment, experiment_name in zip(experiment_dirs, experiment_names):
        for prefix, method in zip(prefix_runs, method_names):
            run_files = glob.glob(f'{experiment}/{prefix}*')
            runs = [pd.read_csv(run)['Value'] for run in run_files]
            
            ewm_runs = [run.ewm(span = ewm_epoch_span).mean() for run in runs] # smooth
            averaged_ewm_runs = pd.concat(ewm_runs, axis = 1).mean(axis=1) # average over folds

            averaged_ewm_runs = pd.DataFrame(averaged_ewm_runs, columns = ['loss'])
            averaged_ewm_runs['iteration'] = pd.read_csv(run_files[0])['Step']
            averaged_ewm_runs['experiment'] = experiment_name
            averaged_ewm_runs['Method'] = method
            averaged_runs = pd.concat([averaged_runs, averaged_ewm_runs])

    sns.set(style="darkgrid")
    sns.relplot(
        data = averaged_runs, 
        x = "iteration", 
        y = "loss", 
        hue = "Method",
        style = "Method",
        col = col, 
        kind = "line",
    )
    plt.ylim(5.4, 7)
    plt.savefig(f'plots/{fig_title}')
    plt.show()

if __name__ == '__main__':
    fire.Fire(plot)
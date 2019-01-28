import sys
from . import utils
import os
import glob
import json
import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description="Get the result statistics of the preprocessing experiments"
    )
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        help="Root directory of experiments"
    )
    return parser.parse_args()


def main():
    args = get_args()
    # For each exp settings 
    root_folder = args.data
    experiment_folders = [
        os.path.join(root_folder, p) for p in os.listdir(root_folder)
    ]
    for setting_folder in experiment_folders:
        experiment_subfolders = glob.glob(setting_folder + "/seed*")
        n = len(experiment_subfolders) # Number of seeds
        results = {}
        models = ['lstm', 'dan']
        tests = ['conala', 'github']
        metrics = ['mrr', 'success@1', 'success@5', 'success@10']
        for model in models:
            results[model] = {}
            for test in tests:
                results[model][test] = {}
                for metric in metrics:
                    results[model][test][metric] = []
        
        for seed_folder in experiment_subfolders:
            # Read results
            for model in models:
                with open(os.path.join(seed_folder, model, 'results.json'),'r') as f:
                    data = json.load(f)
                    for data_entry in data:
                        for metric in metrics:
                            results[data_entry['model']][data_entry['dataset']][metric].append(data_entry[metric])

        # Compute confidence interval
        confidence = 0.95
        results['confidence'] = confidence
        for model in models:
            for test in tests:
                for metric in metrics:
                    results[model][test][metric] = utils.mean_confidence_interval(results[model][test][metric], confidence=confidence)
        
        # Store results under setting root folder
        results_path = os.path.join(setting_folder, "stats.json")
        with open(results_path, "w") as fout:
            json.dump(results, fout)
        print('Finish compute for ' + setting_folder)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()

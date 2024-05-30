import argparse
import pandas as pd
import os
import fnmatch
import numpy as np


def find_files(path, patterns):
    matches = []
    dataframes = {}
    for pattern in patterns:
        dfs = []
        for root, dirnames, filenames in os.walk(path):
            for filename in fnmatch.filter(filenames, f'*{pattern}*'):
                full_path = os.path.join(root, filename)
                matches.append(full_path)
                df = pd.read_csv(full_path)
                dfs.append(df)
        # Concatenate all dataframes for the current pattern
        dataframes[pattern] = pd.concat(dfs, ignore_index=True)
    return matches, dataframes


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('path', type=str, help='Path to search for files')
    args = parser.parse_args()

    patterns = ['_easyrefs', '_medrefs', '_hardrefs']
    matches, dataframes = find_files(args.path, patterns)
    
    for pattern in patterns:
        # Compute the mean of 'avg_rmse' and 'avg_fcs_fluct'
        avg_rmse_mean = dataframes[pattern]['avg_rmse'].mean()
        avg_fcs_fluct_mean = dataframes[pattern]['avg_fcs_fluct'].mean()

        # Compute the standard error of the mean for 'avg_rmse' and 'avg_fcs_fluct'
        avg_rmse_sem = dataframes[pattern]['avg_rmse'].std() / np.sqrt(len(dataframes[pattern]))
        avg_fcs_fluct_sem = dataframes[pattern]['avg_fcs_fluct'].std() / np.sqrt(len(dataframes[pattern]))

        # Append the means and SEMs to the DataFrame
        dataframes[pattern] = dataframes[pattern]._append({
            'avg_rmse': avg_rmse_mean, 
            'avg_rmse_sem': avg_rmse_sem,
            'avg_fcs_fluct': avg_fcs_fluct_mean,
            'avg_fcs_fluct_sem': avg_fcs_fluct_sem
        }, ignore_index=True)

        # Create a new list of column names that includes all the original columns in the desired order
        column_order = [col for col in dataframes[pattern].columns \
                        if col not in ['avg_rmse', 'avg_rmse_sem', 'avg_fcs_fluct', 'avg_fcs_fluct_sem']] + \
                        ['avg_rmse', 'avg_rmse_sem', 'avg_fcs_fluct', 'avg_fcs_fluct_sem']

        # Rearrange the columns
        dataframes[pattern] = dataframes[pattern][column_order]

        # Save the DataFrame to a CSV file, overwriting any existing file
        output_path = os.path.join(args.path, f'{pattern[1:]}_total.csv')
        print(dataframes[pattern])
        dataframes[pattern].to_csv(output_path, index=False)

if __name__ == "__main__":
    main()

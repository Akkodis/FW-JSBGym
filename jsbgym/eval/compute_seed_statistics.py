import argparse
import pandas as pd
import os
import fnmatch
import numpy as np
pd.options.mode.chained_assignment = None  # default='warn'

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
    parser.add_argument('--path', type=str, help='Path to search for files')
    parser.add_argument('--max-ref-dif', type=str, default='hardrefs', help='Maximum reference difficulty up to which to compute statistics',
                        choices=['easyrefs', 'medrefs', 'hardrefs'])
    parser.add_argument('--max-atmo-severity', type=str, default='severe', help='Maximum atmosphere severity up to which to compute statistics',
                        choices=['off', 'light', 'moderate', 'severe'])

    args = parser.parse_args()

    patterns = ['_easyrefs', '_medrefs', '_hardrefs']
    atmo_severities = ['off', 'light', 'moderate', 'severe']

    # copy the patterns list up to the args.max_ref_dif argument
    patterns = patterns[:patterns.index(f'_{args.max_ref_dif}') + 1]
    atmo_severities = atmo_severities[:atmo_severities.index(args.max_atmo_severity) + 1]

    matches, dataframes = find_files(args.path, patterns)
    all_atmo_stats = pd.DataFrame(columns=['ref_lvl','severity', 'roll_rmse', 'pitch_rmse',
                                           'roll_fcs_fluct', 'pitch_fcs_fluct',
                                           'avg_rmse', 'avg_rmse_sem',
                                           'avg_fcs_fluct', 'avg_fcs_fluct_sem'])
    for ref_lvl in patterns:
        for atmo_severity in atmo_severities:
            # Filter the DataFrame to only include rows with the desired atmosphere severity
            per_atmo_sev_df = dataframes[ref_lvl][dataframes[ref_lvl]['severity'] == atmo_severity]
            per_atmo_sev_df['ref_lvl'] = ref_lvl[1:]
            all_atmo_stats = pd.concat([all_atmo_stats, per_atmo_sev_df], ignore_index=True)

            # Compute the mean of 'avg_rmse' and 'avg_fcs_fluct'
            avg_rmse_mean = per_atmo_sev_df['avg_rmse'].mean()
            avg_fcs_fluct_mean = per_atmo_sev_df['avg_fcs_fluct'].mean()

            # Compute the standard error of the mean for 'avg_rmse' and 'avg_fcs_fluct'
            avg_rmse_sem = per_atmo_sev_df['avg_rmse'].std() / np.sqrt(len(per_atmo_sev_df))
            avg_fcs_fluct_sem = per_atmo_sev_df['avg_fcs_fluct'].std() / np.sqrt(len(per_atmo_sev_df))

            # Append the means and SEMs to the DataFrame
            statistics_df = pd.DataFrame([[ref_lvl[1:], atmo_severity, avg_rmse_mean, avg_rmse_sem, avg_fcs_fluct_mean, avg_fcs_fluct_sem]],
                                         columns=['ref_lvl', 'severity', 'avg_rmse', 'avg_rmse_sem', 'avg_fcs_fluct', 'avg_fcs_fluct_sem'])
            all_atmo_stats = pd.concat([all_atmo_stats, statistics_df], ignore_index=True)

            # Save the DataFrame to a CSV file, overwriting any existing file
            output_path = os.path.join(args.path, 'all_total.csv')
            # print(dataframes[ref_lvl])
            all_atmo_stats.to_csv(output_path, index=False)

if __name__ == "__main__":
    main()

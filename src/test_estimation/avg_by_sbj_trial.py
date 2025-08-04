import pandas as pd
import os


def group_save_file(file_path, out_file_path):
    print('filepath: {}'.format(file_path))
    df = pd.read_csv(file_path)
    grouped_df = df.groupby(["Subject", "Speed"]).mean().reset_index()

    grouped_df.to_csv(out_file_path, index=False)

if __name__ == '__main__':
    from configs.configs import parse_configs
    configs = parse_configs()
    results_dir = os.path.join(configs.results_dir, configs.saved_fn)

    speed = 'F'

    file_path = os.path.join(results_dir, f'stride_wise_{speed}.csv')
    out_file_path = os.path.join(results_dir, f'subject_wise_{speed}.csv')

    group_save_file(file_path, out_file_path)

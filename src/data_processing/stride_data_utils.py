import os
import numpy as np
import pandas as pd


def remove_nan(lst):
    # Remove nan values from a list
    return [x for x in lst if not np.isnan(x)]


def get_events_infor(annos_events_path, augment=True):
    """
    Read the gait events CSV and return a list of tuples:
        (start_index, end_index, label)
    where label is 'L' if the cycle starts with a left heel-strike and 'R' if with a right heel-strike.
    The "primary" pairs are taken from the first detected event type.
    If augment is True, we also create offset pairs using the opposite leg.
    """
    events_df = pd.read_csv(annos_events_path)
    left_HS = remove_nan(events_df['LHS'].tolist())
    right_HS = remove_nan(events_df['RHS'].tolist())

    pairs = []
    # Determine which event occurs first
    if len(left_HS) > 0 and len(right_HS) > 0:
        if left_HS[0] < right_HS[0]:
            start = 'L'
        else:
            start = 'R'
    else:
        return pairs  # no valid events

    # Primary pairs: use the foot that occurred first
    if start == 'L':
        # Build pairs: left->right; e.g., from left_HS[i] to right_HS[i+1]
        for i in range(len(left_HS) - 1):
            if i + 1 < len(right_HS):
                pairs.append((int(left_HS[i]), int(right_HS[i + 1]), 'L', 'No'))
    elif start == 'R':
        # Build pairs: right->left; e.g., from right_HS[i] to left_HS[i+1]
        for i in range(len(right_HS) - 1):
            if i + 1 < len(left_HS):
                pairs.append((int(right_HS[i]), int(left_HS[i + 1]), 'R', 'No'))

    # Augmented pairs: slide one full stride (using the opposite leg as start)
    if augment:
        if start == 'L':
            # Additional pairs starting with right heel strike:
            for i in range(len(right_HS) - 2):
                if i + 2 < len(left_HS):
                    pairs.append((int(right_HS[i]), int(left_HS[i + 2]), 'R', 'Yes'))
        elif start == 'R':
            # Additional pairs starting with left heel strike:
            for i in range(len(left_HS) - 2):
                if i + 2 < len(right_HS):
                    pairs.append((int(left_HS[i]), int(right_HS[i + 2]), 'L', 'Yes'))

    return pairs


def get_spatial_params(annos_spatial_path):
    """
    Reads spatial parameter CSV and returns eight lists.
    (These lists should be ordered by gait cycle. Depending on how your CSV is built,
    they may already combine cycles for both feet.)
    """
    df = pd.read_csv(annos_spatial_path)

    # Get the spatial parameters columns
    r_stride = df['Right_StrideLen'].tolist()
    l_stride = df['Left_StrideLen'].tolist()
    r_step = df['Right_StepLen'].tolist()
    l_step = df['Left_StepLen'].tolist()
    r_stride_width = df['Right_StrideWidth'].tolist()
    l_stride_width = df['Left_StrideWidth'].tolist()
    r_step_width = df['Right_StepWidth'].tolist()
    l_step_width = df['Left_StepWidth'].tolist()

    # Remove the first value (if needed to align indices)
    r_stride = r_stride[1:]
    l_stride = l_stride[1:]
    r_step = r_step[1:]
    l_step = l_step[1:]
    r_stride_width = r_stride_width[1:]
    l_stride_width = l_stride_width[1:]
    r_step_width = r_step_width[1:]
    l_step_width = l_step_width[1:]

    # Remove any NaN values from each list
    r_stride = remove_nan(r_stride)
    l_stride = remove_nan(l_stride)
    r_step = remove_nan(r_step)
    l_step = remove_nan(l_step)
    r_stride_width = remove_nan(r_stride_width)
    l_stride_width = remove_nan(l_stride_width)
    r_step_width = remove_nan(r_step_width)
    l_step_width = remove_nan(l_step_width)

    return r_stride, l_stride, r_step, l_step, r_stride_width, l_stride_width, r_step_width, l_step_width


def get_input_sequence(file_path):
    df = pd.read_csv(file_path)

    # Select columns 1 to 6 and 15 to 20 (0-indexed slicing)
    imu_seq1 = df.iloc[:, 1:7].values
    imu_seq2 = df.iloc[:, 15:21].values

    # Concatenate sequences along the feature axis
    input_imu_sequence = np.concatenate((imu_seq1, imu_seq2), axis=1)

    # Remove rows with any NaN values
    input_imu_sequence = input_imu_sequence[~pd.isnull(input_imu_sequence).any(axis=1)]

    return input_imu_sequence


def get_annos_infor(dataset_dir, mode, configs, augment=True):
    """
    Loads data for every subject/trial and splits the IMU sequence using gait event information.
    If 'augment' is True the window will be shifted by one stride (opposite leg) to increase the number of samples.
    """
    annos_infor = []

    input_data_dir = os.path.join(dataset_dir, 'insoles_data')
    target_spatial = os.path.join(dataset_dir, 'target_spatial_parameters')
    target_gait_event = os.path.join(dataset_dir, 'gait_events')


    subjects = sorted(os.listdir(input_data_dir))  # List of subjects
    speeds = configs.test.speeds[0]                                      # N or F
    trials = ['1', '2', '3', '4']

    # Loop over subjects and trials
    for sbj in subjects:
        for speed in speeds:
            for trial in trials:

                input_file_path = os.path.join(input_data_dir, sbj, f'{sbj}_{speed}_{trial}.csv')
                if not os.path.exists(input_file_path):
                    continue

                target_spatial_file_path = os.path.join(target_spatial, sbj, f'{sbj}_{speed}_{trial}.csv')
                if not os.path.exists(target_spatial_file_path):
                    continue

                target_gait_event_file_path = os.path.join(target_gait_event, sbj, f'{sbj}_{speed}_{trial}.csv')
                if not os.path.exists(target_gait_event_file_path):
                    continue

                # Reformat subject ID if needed
                sbj_d, visit = sbj.split('_')
                sbj_d = f'{int(sbj_d):08d}'
                sbj = f'{sbj_d}_{visit}'

                # Load the IMU sequence
                imu_sequence = get_input_sequence(input_file_path)

                # Load spatial parameters (assumed to correspond to the gait cycles)
                r_stride, l_stride, r_step, l_step, r_stride_width, l_stride_width, r_step_width, l_step_width = get_spatial_params(
                    target_spatial_file_path)

                # Get event pairs; note we pass augment=True if we want extra samples.
                pairs = get_events_infor(target_gait_event_file_path, augment=augment)

                # To assign the correct spatial measurement (if available) from the CSV, we
                # use separate counters for cycles starting with left vs. right.
                idx_L = 0
                idx_R = 0

                for (start_idx, end_idx, label, augmented) in pairs:
                    sub_imu_sequence = imu_sequence[start_idx:end_idx]
                    if len(sub_imu_sequence) > 256:
                        continue

                    # For demonstration, we assume that the spatial CSV has rows corresponding
                    # separately to cycles that start with each foot.
                    if label == 'L':
                        if idx_L >= len(r_stride):
                            continue
                        # For cycles starting with left (i.e. primary if file starts with L,
                        # or offset if file starts with R), assign spatial parameters.

                        if augmented == 'Yes':
                            sub_r_stride = r_stride[idx_L + 1]
                            sub_l_stride = l_stride[idx_L]
                            sub_r_step = r_step[idx_L + 1]
                            sub_l_step = l_step[idx_L]
                            sub_r_stride_width = r_stride_width[idx_L + 1]
                            sub_l_stride_width = l_stride_width[idx_L]
                            sub_r_step_width = r_step_width[idx_L + 1]
                            sub_l_step_width = l_step_width[idx_L]
                        else:
                            sub_r_stride = r_stride[idx_L]
                            sub_l_stride = l_stride[idx_L]
                            sub_r_step = r_step[idx_L]
                            sub_l_step = l_step[idx_L]
                            sub_r_stride_width = r_stride_width[idx_L]
                            sub_l_stride_width = l_stride_width[idx_L]
                            sub_r_step_width = r_step_width[idx_L]
                            sub_l_step_width = l_step_width[idx_L]
                        idx_L += 1
                    else:  # label == 'R'
                        if idx_R >= len(r_stride):
                            continue

                        if augmented == 'Yes':
                            sub_r_stride = r_stride[idx_R + 1]
                            sub_l_stride = l_stride[idx_R]
                            sub_r_step = r_step[idx_R + 1]
                            sub_l_step = l_step[idx_R]
                            sub_r_stride_width = r_stride_width[idx_R + 1]
                            sub_l_stride_width = l_stride_width[idx_R]
                            sub_r_step_width = r_step_width[idx_R + 1]
                            sub_l_step_width = l_step_width[idx_R]
                        else:
                            sub_r_stride = r_stride[idx_R]
                            sub_l_stride = l_stride[idx_R]
                            sub_r_step = r_step[idx_R]
                            sub_l_step = l_step[idx_R]
                            sub_r_stride_width = r_stride_width[idx_R]
                            sub_l_stride_width = l_stride_width[idx_R]
                            sub_r_step_width = r_step_width[idx_R]
                            sub_l_step_width = l_step_width[idx_R]

                        idx_R += 1

                    if sub_r_stride_width == 0. or sub_l_stride_width == 0.:
                        # Skip if stride width is zero, which may indicate missing data
                        continue

                    long_spatial_params = [
                        sub_r_stride, sub_l_stride, sub_r_step, sub_l_step,
                        sub_r_step_width, sub_l_step_width
                    ]
                    short_spatial_params = [sub_r_stride_width, sub_l_stride_width]

                    annos_infor.append([sub_imu_sequence, long_spatial_params, short_spatial_params, sbj, speed, trial])
    return annos_infor

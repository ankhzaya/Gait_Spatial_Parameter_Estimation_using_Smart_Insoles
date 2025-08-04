#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# lookup titles for spatial parameters
title_dict_spatial = {
    'RSTRL': 'Right Stride Length',  'LSTRL': 'Left Stride Length',
    'RSTPL': 'Right Step Length',    'LSTPL': 'Left Step Length',
    'RSTRW': 'Right Stride Width',   'LSTRW': 'Left Stride Width',
    'RSTPW': 'Right Step Width',     'LSTPW': 'Left Step Width',
    'STRL':  'Stride Length',        'STPL':  'Step Length',
    'STRW':  'Stride Width',         'STPW':  'Step Width'
}

# colors & labels for normal (N) and fast (F) walking
color_speed = { 'N': '#d31d0d', 'F': '#00487a' }
label_speed = { 'N': 'Usual',    'F': 'Fast'    }

tick_X_dict = {'STRW': [6, 10, 14, 18, 22, 26],}

tick_Y_dict = {'STRL': [-12, -8, -4, 0, 4, 8, 12],
               'STPL': [-9, -6, -3, 0, 3, 6, 9],
               'STRW': [-6, -4, -2, 0, 2, 4, 6],
               'STPW': [-9, -6, -3, 0, 3, 6, 9]}


def create_combined_bland_altman_plot(file_paths, conditions, col_pair, save_dir):
    target_col, pred_col, key = col_pair
    metrics = {}

    # First, gather all diffs so we know the full data range:
    all_diffs = []
    for fp in file_paths:
        df = pd.read_csv(fp)
        all_diffs.append(df[target_col].values - df[pred_col].values)
    diffs      = np.hstack(all_diffs)
    y_min, y_max = diffs.min(), diffs.max()
    y_range     = y_max - y_min
    offset      = 0.015 * y_range   # 2% of the span

    fig, ax = plt.subplots(figsize=(6,6))

    x_pos = 0.98  # 98% from left in axis‐fraction coords
    for fp, cond in zip(file_paths, conditions):
        df     = pd.read_csv(fp)
        target = df[target_col].values
        pred   = df[pred_col].values
        mean_v = (target + pred) / 2.0
        diff_v = target - pred

        # compute BA stats
        bias    = diff_v.mean()
        sd_diff = diff_v.std(ddof=1)
        loa_up  = bias + 1.96 * sd_diff
        loa_lo  = bias - 1.96 * sd_diff
        metrics[cond] = (bias, loa_lo, loa_up)

        # scatter + lines
        ax.scatter(
            mean_v, diff_v,
            color=color_speed[cond],
            alpha=0.7, s=200,
            edgecolors='white', linewidth=0.8,
            label=label_speed[cond]
        )
        ax.axhline(bias,   color=color_speed[cond], linestyle='--', linewidth=1)
        ax.axhline(loa_up, color=color_speed[cond], linestyle=':',  linewidth=1)
        ax.axhline(loa_lo, color=color_speed[cond], linestyle=':',  linewidth=1)

        # choose above vs below offset
        if cond == 'F':
            dy = +offset
            va = 'bottom'
        else:
            dy = -offset
            va = 'top'

        # annotate each line, clamped inside [y_min+offset, y_max-offset]
        for y_val in (bias, loa_up, loa_lo):
            y_text = y_val + dy
            # clamp
            y_text = max(y_min + offset, min(y_text, y_max - offset))
            ax.text(
                x_pos, y_text,
                f"{y_val:.2f}",
                transform=ax.get_yaxis_transform(),
                ha='right', va=va,
                fontsize=10, color=color_speed[cond], alpha=1.0
            )

    # final styling
    TITLE = title_dict_spatial.get(key, key)
    ax.set_title(f'{TITLE}', fontweight='bold', fontsize=17)
    ax.set_xlabel('Mean of True & Predicted (cm)', fontsize=14)
    ax.set_ylabel('Difference (True – Pred) (cm)', fontsize=14)
    ax.grid(alpha=0.3, linestyle=':', linewidth=0.5)
    ax.tick_params(direction='in')
    ax.legend(loc='lower right', fontsize=11)

    # set x-ticks if available4
    if key in tick_X_dict:
        ax.set_xticks(tick_X_dict[key])
        ax.set_xticklabels(tick_X_dict[key], fontsize=12)
    else:
        ax.tick_params(axis='x', labelsize=12)

    # set y-ticks if available
    # if key in tick_Y_dict:
    #     ax.set_yticks(tick_Y_dict[key])
    #     ax.set_yticklabels(tick_Y_dict[key], fontsize=12)
    # else:
    #     ax.tick_params(axis='y', labelsize=12)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{key}_bland_altman_combined.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return metrics


if __name__ == '__main__':
    from configs.configs import parse_configs
    configs     = parse_configs()
    results_dir = os.path.join(configs.results_dir, configs.saved_fn)

    evaluation = 'stride'
    # evaluation = 'subject'
    if evaluation == 'stride':
        normal_csv  = os.path.join(results_dir, 'stride_wise_N.csv')
        fast_csv    = os.path.join(results_dir, 'stride_wise_F.csv')
    else:
        # subject-wise evaluation
        normal_csv  = os.path.join(results_dir, f'subject_wise_N.csv')
        fast_csv    = os.path.join(results_dir, f'subject_wise_F.csv')

    save_dir    = os.path.join(results_dir, f'combined_bland_altman_plots_{evaluation}')
    file_paths = [normal_csv, fast_csv]
    conditions = ['N', 'F']

    spat_col_pairs = [
        ('Target_STRL', 'Pred_STRL', 'STRL'),
        ('Target_STPL', 'Pred_STPL', 'STPL'),
        ('Target_STRW', 'Pred_STRW', 'STRW'),
        ('Target_STPW', 'Pred_STPW', 'STPW'),
    ]

    for pair in spat_col_pairs:
        metrics = create_combined_bland_altman_plot(file_paths, conditions, pair, save_dir)
        print(pair[2], "→", metrics)
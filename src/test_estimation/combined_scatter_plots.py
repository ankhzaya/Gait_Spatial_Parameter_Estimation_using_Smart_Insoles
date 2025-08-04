#!/usr/bin/env python3
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pingouin as pg
import warnings

warnings.filterwarnings("ignore")

# === Title lookup for spatial parameters ===
title_dict_spatial = {
    'RSTRL': 'Right Stride Length',
    'LSTRL': 'Left Stride Length',
    'RSTPL': 'Right Step Length',
    'LSTPL': 'Left Step Length',
    'RSTRW': 'Right Stride Width',
    'LSTRW': 'Left Stride Width',
    'RSTPW': 'Right Step Width',
    'LSTPW': 'Left Step Width',
    'STRL':  'Stride Length',
    'STPL':  'Step Length',
    'STRW':  'Stride Width',
    'STPW':  'Step Width'
}

tick_Y_dict = {'STRW': [5, 10, 15, 20, 25],}

def get_ICC(target, pred):
    """
    Compute ICC(2,k) between target and pred lists using pingouin.
    Returns a single float.
    """
    idx = list(range(len(target)))
    judge = ['target'] * len(target) + ['pred'] * len(pred)
    scores = np.concatenate((target, pred))
    df_icc = pd.DataFrame({
        'index': idx + idx,
        'judge': judge,
        'scores': scores
    })
    icc_res = pg.intraclass_corr(data=df_icc, targets='index', raters='judge', ratings='scores')
    icc_val = icc_res.loc[icc_res['Type']=='ICC2k', 'ICC'].values[0]
    return icc_val

def plot_by_speed(df, target_col, pred_col, short_name, save_dir):
    """
    Create combined lmplot for two walking speeds (N, F), compute/annotate overall
    and per-speed r & ICC in the legend.
    """
    # overall metrics
    r_all, _   = stats.pearsonr(df[pred_col], df[target_col])
    icc_all    = get_ICC(df[target_col].tolist(), df[pred_col].tolist())

    # per-speed metrics
    speeds = ['N', 'F']
    metrics = {}
    for s in speeds:
        sub = df[df['Condition'] == s]
        r_s, _  = stats.pearsonr(sub[pred_col], sub[target_col])
        icc_s   = get_ICC(sub[target_col].tolist(), sub[pred_col].tolist())
        metrics[s] = (r_s, icc_s)

    # lmplot without regression line
    g = sns.lmplot(
        x=pred_col, y=target_col, data=df,
        hue='Condition', fit_reg=False,
        markers=['o', 'o'],
        palette={'N':'#d31d0d', 'F':'#00487a'},
        height=5, aspect=1, legend=True,
        scatter_kws={'s':130, 'linewidths':0.8, 'edgecolor':'white', 'alpha':0.8}
    )

    # thicken any invisible regression lines
    for line in g.axes[0,0].lines:
        line.set_linewidth(1.2)

    # grab legend
    leg = g.axes[0,0].get_legend()
    if leg is None:
        leg = g._legend

    # set legend title
    leg.set_title(f"Overall r={r_all:.2f}, ICC={icc_all:.2f}\nSpeeds (r, ICC):")

    # update legend labels
    new_labels = [
        f"Usual ({metrics['N'][0]:.2f}, {metrics['N'][1]:.2f})",
        f"Fast   ({metrics['F'][0]:.2f}, {metrics['F'][1]:.2f})"
    ]
    for text, new in zip(leg.texts, new_labels):
        text.set_text(new)
    for text in leg.get_texts():
        text.set_fontsize('large')

    sns.move_legend(g, "lower right", bbox_to_anchor=(.9, .15))

    # title & axes formatting
    title = title_dict_spatial.get(short_name, short_name)
    fig = g.fig
    fig.suptitle(title, fontsize=17, fontweight='bold')
    ax = g.axes[0,0]
    ax.set_xlabel('Predicted (cm)', fontsize=14)
    ax.set_ylabel('True (cm)', fontsize=14)
    ax.tick_params(direction='in')

    # set y-ticks if available
    if short_name in tick_Y_dict:
        ax.set_yticks(tick_Y_dict[short_name])
        ax.set_yticklabels(tick_Y_dict[short_name])
    else:
        ax.tick_params(axis='y', labelsize=12)

    # save figure
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{short_name}_by_speed.png")
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    from configs.configs import parse_configs
    configs = parse_configs()

    results_dir = os.path.join(configs.results_dir, configs.saved_fn)

    # specify the CSV files
    # evaluation = 'stride'
    evaluation = 'subject'

    if evaluation == 'stride':
        normal_csv = os.path.join(results_dir, 'stride_wise_N.csv')
        fast_csv = os.path.join(results_dir, 'stride_wise_F.csv')
    else:
        # subject-wise evaluation
        normal_csv = os.path.join(results_dir, f'subject_wise_N.csv')
        fast_csv = os.path.join(results_dir, f'subject_wise_F.csv')

    save_dir = os.path.join(results_dir, f'combined_scatter_plots_{evaluation}')
    os.makedirs(save_dir, exist_ok=True)

    # load & tag
    df_norm = pd.read_csv(normal_csv)
    df_norm['Condition'] = 'N'
    df_fast = pd.read_csv(fast_csv)
    df_fast['Condition'] = 'F'

    df_all = pd.concat([df_norm, df_fast], ignore_index=True)

    # define your column pairs
    spat_col_pairs = [
        ('Target_STRL', 'Pred_STRL', 'STRL'),
        ('Target_STPL', 'Pred_STPL', 'STPL'),
        ('Target_STRW', 'Pred_STRW', 'STRW'),
        ('Target_STPW', 'Pred_STPW', 'STPW'),
    ]

    # generate plots
    for tgt, prd, short in spat_col_pairs:
        df = df_all[[tgt, prd, 'Condition']].dropna()
        plot_by_speed(df, tgt, prd, short, save_dir)
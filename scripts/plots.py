"""Generate all README figures from saved artifacts.

Requires models.py to have been run first so that:
  - export/oof_preds.csv
  - export/feature_importance.csv
exist alongside the cleaned data.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

SRC  = r'F:\housing_prices\source'
EXP  = r'F:\housing_prices\export'
OUT  = r'F:\housing_prices\plots\performance'


# -----------------------------------------------------------------------------
# Figure 1 — target distribution before and after log1p
# -----------------------------------------------------------------------------
def plot_target_distribution():
    df = pd.read_csv(rf'{SRC}\train.csv')
    sp     = df['SalePrice']
    sp_log = np.log1p(sp)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].hist(sp / 1000, bins=60, color='#5b8def', edgecolor='white')
    axes[0].set_title(f'Raw SalePrice  (skew = {sp.skew():.2f})')
    axes[0].set_xlabel('SalePrice ($thousands)')
    axes[0].set_ylabel('Count')

    axes[1].hist(sp_log, bins=60, color='#3aa776', edgecolor='white')
    axes[1].set_title(f'log1p(SalePrice)  (skew = {sp_log.skew():.2f})')
    axes[1].set_xlabel('log1p(SalePrice)')
    axes[1].set_ylabel('Count')

    fig.suptitle('Why we log-transform the target', y=1.02, fontsize=13)
    plt.tight_layout()
    plt.savefig(rf'{OUT}\target_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()


# -----------------------------------------------------------------------------
# Figure 2 — RMSE comparison, prior pipeline vs current
# -----------------------------------------------------------------------------
def plot_rmse_comparison():
    rows = [
        # (label,            prior,    current)
        ('Linear',           0.12672,  0.12234),
        ('Ridge',            0.11126,  0.11003),
        ('Lasso',            0.10988,  0.10818),
        ('ElasticNet',       np.nan,   0.10817),
        ('KernelRidge',      np.nan,   0.11634),
        ('XGBoost',          0.11783,  0.10940),
        ('LightGBM',         np.nan,   0.11208),
        ('NNLS Blend',       0.10809,  0.10534),
        ('Stacked Ridge',    0.10786,  0.10544),
        ('Final Ensemble',   np.nan,   0.10537),
    ]
    labels  = [r[0] for r in rows]
    prior   = np.array([r[1] for r in rows], dtype=float)
    current = np.array([r[2] for r in rows], dtype=float)

    x = np.arange(len(labels))
    w = 0.38

    fig, ax = plt.subplots(figsize=(11, 5))
    bars_prior = ax.bar(x - w/2, np.where(np.isnan(prior), 0, prior), w,
                        label='Previous pipeline', color='#b6b6b6', edgecolor='white')
    bars_curr  = ax.bar(x + w/2, current, w,
                        label='Current pipeline',  color='#5b8def', edgecolor='white')

    # mark "new in current" bars with hatching on the prior side
    for i, p in enumerate(prior):
        if np.isnan(p):
            bars_prior[i].set_alpha(0)

    # annotate current bars with their value
    for i, v in enumerate(current):
        ax.text(x[i] + w/2, v + 0.0008, f'{v:.4f}',
                ha='center', va='bottom', fontsize=8, color='#1f3a8a')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha='right')
    ax.set_ylabel('Out-of-fold RMSE on log1p(SalePrice)')
    ax.set_title('OOF RMSE: previous pipeline vs current')
    ax.set_ylim(0.10, max(current.max(), np.nan_to_num(prior).max()) + 0.005)
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(rf'{OUT}\rmse_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


# -----------------------------------------------------------------------------
# Figure 3 — OOF prediction correlations between base models
# -----------------------------------------------------------------------------
def plot_oof_correlation():
    oof = pd.read_csv(rf'{EXP}\oof_preds.csv')
    cols = ['ridge', 'lasso', 'enet', 'kridge', 'xgb', 'lgb']
    pretty = ['Ridge', 'Lasso', 'ElasticNet', 'KernelRidge', 'XGBoost', 'LightGBM']
    corr = oof[cols].corr().values

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(corr, cmap='RdYlGn_r', vmin=0.93, vmax=1.0)
    ax.set_xticks(range(len(cols))); ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(pretty, rotation=35, ha='right')
    ax.set_yticklabels(pretty)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f'{corr[i, j]:.3f}', ha='center', va='center',
                    color='white' if corr[i, j] > 0.985 else 'black', fontsize=9)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label('Pearson r')
    ax.set_title('Pairwise correlation of OOF predictions')
    plt.tight_layout()
    plt.savefig(rf'{OUT}\oof_correlation.png', dpi=150, bbox_inches='tight')
    plt.close()


# -----------------------------------------------------------------------------
# Figure 4 — top features by |Lasso coefficient|, signed
# -----------------------------------------------------------------------------
def plot_feature_importance(top_n=20):
    imp = pd.read_csv(rf'{EXP}\feature_importance.csv')
    top = imp.nlargest(top_n, 'lasso_abs').iloc[::-1]   # smallest at top for nicer barh

    colors = ['#cc4f4f' if c < 0 else '#3aa776' for c in top['lasso_signed']]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(top['feature'], top['lasso_signed'], color=colors, edgecolor='white')
    ax.axvline(0, color='black', linewidth=0.6)
    ax.set_xlabel('Lasso coefficient (positive = raises predicted log-price)')
    ax.set_title(f'Top {top_n} features by |Lasso coefficient|')
    handles = [Patch(facecolor='#3aa776', label='positive'),
               Patch(facecolor='#cc4f4f', label='negative')]
    ax.legend(handles=handles, loc='lower right', frameon=False)
    plt.tight_layout()
    plt.savefig(rf'{OUT}\feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()


# -----------------------------------------------------------------------------
# Figure 5 — Ensemble weights (NNLS) for current run
# -----------------------------------------------------------------------------
def plot_ensemble_weights():
    rows = [
        ('Ridge',       0.000, 0.033),
        ('Lasso',       0.000, 0.222),
        ('ElasticNet',  0.472, 0.223),
        ('KernelRidge', 0.099, 0.122),
        ('XGBoost',     0.429, 0.306),
        ('LightGBM',    0.000, 0.102),
    ]
    labels = [r[0] for r in rows]
    nnls   = [r[1] for r in rows]
    meta   = [r[2] for r in rows]

    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w/2, nnls, w, label='NNLS blend',         color='#5b8def', edgecolor='white')
    ax.bar(x + w/2, meta, w, label='Ridge meta (coef.)', color='#f0a050', edgecolor='white')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_ylabel('Weight / coefficient')
    ax.set_title('Ensemble weights: NNLS vs Ridge meta')
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(rf'{OUT}\ensemble_weights.png', dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    plot_target_distribution()
    plot_rmse_comparison()
    plot_oof_correlation()
    plot_feature_importance()
    plot_ensemble_weights()
    print('Wrote:')
    for f in ('target_distribution', 'rmse_comparison', 'oof_correlation',
              'feature_importance', 'ensemble_weights'):
        print(f'  plots/performance/{f}.png')

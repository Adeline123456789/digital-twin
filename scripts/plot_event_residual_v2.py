"""Standalone result figures from paired evidence; no raw RGB or assets in Git."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from learn import DATA, ROOT


def save(fig, name):
    destination = ROOT / 'experiments/event_residual_mvp_v2' / (name + '.svg')
    fig.savefig(destination, metadata={'Date': None}, bbox_inches='tight')
    destination.write_text('\n'.join(line.rstrip() for line in destination.read_text().splitlines()) + '\n')
    fig.savefig(ROOT / '.local/event_residual_mvp_v2' / (name + '.png'), dpi=160, bbox_inches='tight')
    plt.close(fig)


def main():
    folder = ROOT / 'experiments/event_residual_mvp_v2'
    result = json.loads((folder / 'heldout_results.json').read_text())
    variants = ['B0', 'B1', 'B2_structured', 'B3_structured', 'B3_sparse']
    names = ['Nominal', 'Search', 'Phase PPO', 'History PPO', 'History sparse']
    colors = ['#777777', '#cc9a30', '#7c64a2', '#3277a5', '#648c68']
    rows = [result['statistics'][v]['heldout'] for v in variants]
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    rates = [r['success_rate'] * 100 for r in rows]
    axes[0].barh(names, rates, color=colors)
    axes[0].invert_yaxis(); axes[0].set_xlim(0, 100); axes[0].set_xlabel('Success (%)')
    for i, r in enumerate(rows): axes[0].text(rates[i] + 1, i, f"{r['successes']}/{r['n']}", va='center')
    axes[0].set_title('Local insertion: 32 held-out pairs')
    for i, r in enumerate(rows[1:]):
        lo, hi = r['ci95_pp']; mean = r['improvement_pp']
        axes[1].errorbar(mean, i, xerr=[[mean-lo], [hi-mean]], fmt='o', color=colors[i+1], capsize=4)
    axes[1].axvline(0, color='#888888', linestyle='--', lw=1)
    axes[1].set_yticks(range(4), names[1:]); axes[1].invert_yaxis()
    axes[1].set_xlabel('Improvement over nominal (percentage points)')
    axes[1].set_title('Paired bootstrap 95% interval')
    save(fig, 'paired_recovery')
    nominal = {r['case_id']:r for r in json.loads((folder/'evaluation_B0_dropout0_recheck.json').read_text())['cases']}
    learned = json.loads((folder/'evaluation_B3_structured_dropout0.json').read_text())['cases']
    chosen = next(r for r in learned if r['group']=='heldout' and r['is_success'] and not nominal[r['case_id']]['is_success'])
    raw = json.loads((DATA / chosen['trace']['dataset_relative_path']).read_text())
    trace = raw['trace']; time = (np.array([r['frame'] for r in trace]) - 441) / 30
    xy = np.array([r['residual_xy_m'] for r in trace]) * 1000
    velocity = np.array([r['residual_velocity_m_s'] for r in trace]) * 1000
    delta = (np.array([r['final_action'] for r in trace]) - np.array([r['nominal_action'] for r in trace]))[:, 7:13]
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 6), sharex=True, constrained_layout=True)
    axes[0].plot(time, xy[:, 0], label='X'); axes[0].plot(time, xy[:, 1], label='Y')
    axes[0].axhline(2, color='gray', linestyle='--', lw=.8); axes[0].axhline(-2, color='gray', linestyle='--', lw=.8)
    axes[0].set_ylabel('Residual (mm)'); axes[0].legend(ncol=2, frameon=False)
    axes[0].set_title(chosen['case_id'] + ': illustrative recovery; nominal failed')
    axes[1].plot(time, velocity); axes[1].set_ylabel('Residual speed\n(mm/s)'); axes[1].set_ylim(-3.2, 3.2)
    axes[2].plot(time, np.rad2deg(delta), lw=1); axes[2].set_ylabel('Joint correction (deg)')
    axes[2].set_xlabel('Local simulation time (s); recorded at 15 Hz, filter runs at 30 Hz')
    for ax in axes: ax.grid(alpha=.2)
    save(fig, 'smooth_residual_example')
    (folder/'figure_evidence.json').write_text(json.dumps(dict(illustrative_case=chosen['case_id'],
        selection='First predeclared heldout condition recovered by B3 and failed by B0; illustrative, not an independent trial',
        trace=chosen['trace']),indent=2)+'\n')


if __name__ == '__main__':
    main()

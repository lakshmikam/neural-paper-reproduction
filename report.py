"""Build aggregate tables and figures from saved runs; no selective seed reporting."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lagaris import CASES

out = Path('results')
runs = [json.loads(p.read_text()) for p in sorted(out.glob('problem_*.json'))]
with (out/'metrics.csv').open('w', newline='') as f:
    fields = list(dict.fromkeys(k for r in runs for k in r))
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader(); writer.writerows(runs)
paper = {'5': (5e-7, 2e-8, 1.5e-5), '6': (.0015, .0002, .0025),
         '7': (6e-6, 7e-7, 4e-5), '8': (1.5e-5, 6e-7, 4e-5)}
lines = ['# Measured reproduction results', '',
         'All seeds are reported. Errors are absolute solution errors, not residual loss.', '',
         '| Case | Runs | Test max error: min / median / max | Median residual SSE |',
         '|---|---:|---:|---:|']
for c in CASES:
    rr = [r for r in runs if r['case'] == c]
    if not rr: continue
    e = [r['test_max_error'] for r in rr]
    lines.append(f'| {c} | {len(rr)} | {min(e):.3e} / {np.median(e):.3e} / {max(e):.3e} | {np.median([r["final_loss"] for r in rr]):.3e} |')
lines += ['', '## PDE comparison with preprint Table 1', '',
          'Our NN column uses the median of all seeds on 900 cell-center test points. FEM uses 23×23 endpoint-inclusive points; these are distinct evaluation sets.', '',
          '| Problem | Paper NN interpolation | Our NN median | Paper FEM nodes / interpolation | Our FEM nodes / interpolation |',
          '|---|---:|---:|---:|---:|']
for c, (nn, fn, fi) in paper.items():
    rr = [r for r in runs if r['case'] == c]
    path = out/f'fem_{c}.json'
    if not rr or not path.exists(): continue
    fem = json.loads(path.read_text())
    med = np.median([r['test_max_error'] for r in rr])
    lines.append(f'| {c} | {nn:.3e} | {med:.3e} | {fn:.3e} / {fi:.3e} | {fem["node_max_error"]:.3e} / {fem["interpolation_max_error"]:.3e} |')
warnings = [r for r in runs if not r['optimizer_success']]
lines += ['', '## Optimizer outcomes', '',
          f'{len(warnings)} of {len(runs)} runs returned a non-success optimizer status. See per-run JSON for the exact message, iteration count and loss.', '',
          f'Total recorded neural training time: {sum(r["seconds"] for r in runs):.1f} CPU wall-clock seconds.', '',
          '## Interpretation', '',
          'Problem 4 exposes a collocation failure: a tiny residual at ten points does not guarantee an accurate solution between them. The seed-0 run converges in optimizer terms but has a large solution error. This result is retained, not filtered out.', '',
          'BFGS precision-loss warnings can occur even at small solution errors. Accuracy and optimizer termination are reported separately. The difficult oscillatory Problem 6 must be assessed by measured errors, not the architecture alone.', '',
          'This is an independent numerical reproduction of the 1997 preprint, not an exact recreation of the historical software, figures, or random initialization. See README for protocol differences.']
(out/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
fig, axs = plt.subplots(3, 3, figsize=(14, 11))
for ax, c in zip(axs.ravel(), CASES):
    for r in [r for r in runs if r['case'] == c]:
        a = np.load(out/f'problem_{c}_seed_{r["seed"]}.npz')
        ax.semilogy(np.maximum(a['loss_evaluations'], 1e-30), label=f'seed {r["seed"]}', alpha=.8)
    ax.set_title(f'Problem {c}'); ax.set_xlabel('Objective evaluation'); ax.set_ylabel('Residual SSE'); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(out/'training_curves.png', dpi=150); plt.close(fig)
fig, axs = plt.subplots(1, 4, figsize=(16, 3.8))
for ax, c in zip(axs, ('5', '6', '7', '8')):
    path = out/f'problem_{c}_seed_0.npz'
    if not path.exists(): continue
    a = np.load(path)
    im = ax.imshow(a['error'].reshape(30, 30).T, origin='lower', extent=(0, 1, 0, 1), cmap='coolwarm')
    fig.colorbar(im, ax=ax, shrink=.7); ax.set_title(f'Problem {c}, seed 0'); ax.set_xlabel('x'); ax.set_ylabel('y')
fig.tight_layout(); fig.savefig(out/'pde_errors.png', dpi=150); plt.close(fig)
fig, axs = plt.subplots(2, 3, figsize=(14, 7))
for ax, c in zip(axs.ravel(), CASES[:5]):
    for r in [r for r in runs if r['case'] == c]:
        a = np.load(out/f'problem_{c}_seed_{r["seed"]}.npz')
        for component in range(a['error'].shape[1]):
            ax.plot(a['coordinates'][:, 0], a['error'][:, component], label=f'seed {r["seed"]}, output {component+1}', alpha=.8)
    ax.set_title(f'Problem {c}'); ax.set_xlabel('x'); ax.set_ylabel('Signed solution error'); ax.legend(fontsize=7)
axs.ravel()[-1].axis('off')
fig.tight_layout(); fig.savefig(out/'ode_errors.png', dpi=150); plt.close(fig)
print(out/'REPORT.md')

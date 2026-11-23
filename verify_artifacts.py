"""Recompute results from every committed checkpoint, using the reference autograd path."""
import json
from pathlib import Path
import numpy as np
import torch
from lagaris import CASES, Trial, grid, residual, exact

torch.set_num_threads(1)
out = Path('results')
checks = []
for c in CASES:
    for seed in (0, 1, 2):
        stem = out/f'problem_{c}_seed_{seed}'
        meta = json.loads(Path(str(stem)+'.json').read_text())
        saved = torch.load(str(stem)+'.pt', weights_only=True)
        m = Trial(c); m.load_state_dict(saved['state_dict'])
        a = np.load(str(stem)+'.npz')
        z = torch.tensor(a['coordinates'], requires_grad=True)
        pred = m(z)
        np.testing.assert_allclose(pred.detach().numpy(), a['prediction'], atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(exact(c, z).detach().numpy(), a['exact'], atol=1e-12, rtol=1e-12)
        e = float((pred-exact(c, z)).detach().abs().max())
        np.testing.assert_allclose(e, meta['test_max_error'], atol=1e-12, rtol=1e-12)
        r = residual(c, z, pred).detach()
        assert bool(torch.isfinite(r).all())
        trainz = grid(c)
        sse = float(residual(c, trainz, m(trainz)).detach().square().sum())
        np.testing.assert_allclose(sse, meta['final_loss'], atol=1e-12, rtol=1e-6)
        checks.append(dict(case=c, seed=seed, reloaded=True, reference_train_sse=sse,
                           test_residual_rmse=float(r.square().mean().sqrt()),
                           test_max_error=e))
(out/'verification.json').write_text(json.dumps(checks, indent=2)+'\n')
print(f'Verified {len(checks)} checkpoints against independent reference autograd residuals.')

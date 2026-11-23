"""Train every paper case using full-batch float64 BFGS; save auditable artifacts."""
import argparse
import json
import platform
import time
from pathlib import Path
import numpy as np
import scipy
from scipy.optimize import minimize
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from lagaris import CASES, Trial, grid, exact, residual, domain
from fast_residual import make_residual


def train(case, seed, out, maxiter):
    torch.manual_seed(seed)
    model = Trial(case)
    z = grid(case)
    fast = make_residual(model, z) if int(case[0]) >= 5 else None
    history = []
    start = time.perf_counter()

    def objective(w):
        vector_to_parameters(torch.from_numpy(w.copy()), model.parameters())
        model.zero_grad(set_to_none=True)
        z.grad = None
        loss = (fast() if fast else residual(case, z, model(z))).square().sum()
        loss.backward()
        history.append(float(loss.detach()))
        return float(loss.detach()), np.concatenate([p.grad.detach().numpy().ravel() for p in model.parameters()])

    initial = parameters_to_vector(model.parameters()).detach().numpy().copy()
    fit = minimize(objective, initial, jac=True, method='BFGS',
                   options={'maxiter': maxiter, 'gtol': 1e-10})
    vector_to_parameters(torch.from_numpy(fit.x.copy()), model.parameters())
    test = grid(case, 30 if int(case[0]) >= 5 else 1000, offset=True)
    pred = model(test).detach().numpy()
    truth = exact(case, test).detach().numpy()
    error = pred-truth
    train_error = (model(z)-exact(case, z)).detach().numpy()
    stats = dict(case=case, seed=seed, parameters=len(fit.x), iterations=int(fit.nit),
                 evaluations=int(fit.nfev), optimizer_success=bool(fit.success),
                 optimizer_message=str(fit.message), initial_loss=history[0], final_loss=float(fit.fun),
                 train_max_error=float(abs(train_error).max()), test_max_error=float(abs(error).max()),
                 test_rmse=float(np.sqrt(np.mean(error**2))), seconds=time.perf_counter()-start)
    if int(case[0]) < 5:
        ext = torch.linspace(domain(case), domain(case)*1.2, 201)[:, None].requires_grad_(True)
        stats['extrapolation_max_error'] = float((model(ext)-exact(case, ext)).detach().abs().max())
    stem = out/f'problem_{case}_seed_{seed}'
    torch.save({'case': case, 'seed': seed, 'state_dict': model.state_dict()}, str(stem)+'.pt')
    np.savez_compressed(str(stem)+'.npz', coordinates=test.detach().numpy(), prediction=pred,
                        exact=truth, error=error, loss_evaluations=np.array(history))
    Path(str(stem)+'.json').write_text(json.dumps(stats, indent=2)+'\n')
    print(json.dumps(stats), flush=True)
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cases', nargs='+', default=CASES, choices=CASES)
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--maxiter', type=int, default=12000)
    p.add_argument('--out', type=Path, default=Path('results'))
    p.add_argument('--skip-existing', action='store_true', help='Keep completed case/seed artifacts')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    env = {'python': platform.python_version(), 'platform': platform.platform(),
           'torch': torch.__version__, 'numpy': np.__version__, 'scipy': scipy.__version__,
           'device': 'cpu', 'dtype': 'float64', 'threads': 1,
           'pde_residual': 'analytic sigmoid derivatives; cached boundary derivatives',
           'maxiter': args.maxiter, 'seeds': args.seeds, 'cases': args.cases}
    (args.out/'environment.json').write_text(json.dumps(env, indent=2)+'\n')
    for c in args.cases:
        for s in args.seeds:
            if args.skip_existing and (args.out/f'problem_{c}_seed_{s}.json').exists():
                continue
            train(c, s, args.out, args.maxiter)


if __name__ == '__main__': main()

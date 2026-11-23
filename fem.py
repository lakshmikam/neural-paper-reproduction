"""Q2 Galerkin FEM, 18x18 rectangular elements, 3x3 Gauss quadrature."""
import json
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import torch
from lagaris import exact, forcing


def basis(t):
    return np.array([2*(t-.5)*(t-1), 4*t*(1-t), 2*t*(t-.5)]), np.array([4*t-3, 4-8*t, 4*t-1])


def shape(x, y):
    a, da = basis(x)
    b, db = basis(y)
    return np.outer(a, b).ravel(), np.outer(da, b).ravel(), np.outer(a, db).ravel()


def values(case, xy, rhs=False):
    return (forcing if rhs else exact)(case, torch.tensor(xy)).detach().numpy().ravel()


def solve(case, elements=18):
    n = 2*elements+1
    h = 1/elements
    coords = np.array([(x, y) for x in np.linspace(0, 1, n) for y in np.linspace(0, 1, n)])
    ids = np.array([[(2*i+a)*n+2*j+b for a in range(3) for b in range(3)]
                    for i in range(elements) for j in range(elements)])
    origins = np.array([(i*h, j*h) for i in range(elements) for j in range(elements)])
    g, w = leggauss(3)
    g, w = (g+1)/2, w/2
    shapes = [shape(a, b) for a in g for b in g]
    phi = np.array([v[0] for v in shapes])
    dx = np.array([v[1] for v in shapes])/h
    dy = np.array([v[2] for v in shapes])/h
    qw = np.outer(w, w).ravel()*h*h
    qxy = origins[:, None, :]+h*np.array([(a, b) for a in g for b in g])[None, :, :]
    f = values(case, qxy.reshape(-1, 2), True).reshape(-1, 9)
    k = np.einsum('qi,qj,q->ij', dx, dx, qw)+np.einsum('qi,qj,q->ij', dy, dy, qw)
    load = np.einsum('eq,qi,q->ei', f, phi, qw)
    mixed = case in ('7', '8')
    fixed = (coords[:, 0] == 0) | (coords[:, 0] == 1) | (coords[:, 1] == 0)
    if not mixed: fixed |= coords[:, 1] == 1
    free = np.flatnonzero(~fixed)
    u = np.random.default_rng(0).normal(0, .01, len(coords))
    u[fixed] = values(case, coords[fixed])
    if mixed:
        for e in range(elements):
            eid = e*elements+elements-1
            for t, weight in zip(g, w):
                ph, _, _ = shape(t, 1)
                load[eid] -= ph*(2*np.sin(np.pi*(e+t)*h))*weight*h
    rows = np.repeat(ids, 9, axis=1).ravel()
    cols = np.tile(ids, (1, 9)).ravel()
    norms = []
    for _ in range(12):
        local = u[ids]
        r = local@k.T+load
        jac = np.broadcast_to(k, (len(ids), 9, 9)).copy()
        if case == '8':
            uq, uy = local@phi.T, local@dy.T
            r -= np.einsum('eq,eq,qi,q->ei', uq, uy, phi, qw)
            jac -= np.einsum('eq,qi,qj,q->eij', uy, phi, phi, qw)
            jac -= np.einsum('eq,qi,qj,q->eij', uq, phi, dy, qw)
        total = np.zeros(len(coords))
        np.add.at(total, ids.ravel(), r.ravel())
        norms.append(float(np.linalg.norm(total[free])))
        if norms[-1] < 1e-11: break
        mat = coo_matrix((jac.ravel(), (rows, cols)), shape=(len(coords), len(coords))).tocsr()
        u[free] += spsolve(mat[free][:, free], -total[free])
    if norms[-1] > 1e-9: raise RuntimeError(f'FEM {case} failed: {norms}')
    test = np.array([(x, y) for x in np.linspace(0, 1, 23) for y in np.linspace(0, 1, 23)])
    interpolated = []
    for x, y in test:
        i, j = min(int(x*elements), elements-1), min(int(y*elements), elements-1)
        ph, _, _ = shape(x*elements-i, y*elements-j)
        interpolated.append(ph@u[ids[i*elements+j]])
    stats = dict(case=case, elements_per_axis=elements, nodes=len(coords), free_unknowns=len(free),
                 residual_norms=norms, node_max_error=float(abs(u-values(case, coords)).max()),
                 interpolation_max_error=float(abs(np.array(interpolated)-values(case, test)).max()))
    return stats, coords, u, test, np.array(interpolated)


if __name__ == '__main__':
    out = Path('results'); out.mkdir(exist_ok=True)
    for c in ('5', '6', '7', '8'):
        stats, coords, u, test, pred = solve(c)
        (out/f'fem_{c}.json').write_text(json.dumps(stats, indent=2)+'\n')
        np.savez_compressed(out/f'fem_{c}.npz', coordinates=coords, prediction=u, test_coordinates=test, test_prediction=pred)
        print(json.dumps(stats), flush=True)

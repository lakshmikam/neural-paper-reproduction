"""Equivalent PDE residual with analytic sigmoid input derivatives and cached BCs."""
import math
import torch
from lagaris import derivative, forcing


def network_derivatives(net, z):
    s = torch.sigmoid(net[0](z))
    v = net[2].weight.T
    wx, wy = net[0].weight[:, 0], net[0].weight[:, 1]
    s1 = s*(1-s)
    s2 = s1*(1-2*s)
    s3 = s1*(1-6*s+6*s*s)
    return (s@v, (s1*wx)@v, (s1*wy)@v, (s2*wx**2)@v,
            (s2*wy**2)@v, (s2*wx*wy)@v, (s3*wx**2*wy)@v)


def make_residual(model, z):
    c = model.case
    x, y = z.detach()[:, :1], z.detach()[:, 1:2]
    rhs = forcing(c, z).detach()
    mixed = c in ('7', '8')
    if mixed:
        b = 2*y*torch.sin(math.pi*x)
        lapb = -math.pi**2*b
        by = 2*torch.sin(math.pi*x)
        C, Cx, Cy = x*(1-x)*y, (1-2*x)*y, x*(1-x)
        Cl = -2*y
    else:
        C = x*(1-x)*y*(1-y)
        Cx, Cy = (1-2*x)*y*(1-y), x*(1-x)*(1-2*y)
        Cl = -2*(y*(1-y)+x*(1-x))
        # Cancel the trainable correction before taking fixed boundary derivatives.
        zg = z.detach().clone().requires_grad_(True)
        a = model(zg)-zg[:, :1]*(1-zg[:, :1])*zg[:, 1:2]*(1-zg[:, 1:2])*model.nets[0](zg)
        lapb = (derivative(derivative(a, zg), zg)+derivative(derivative(a, zg, 1), zg, 1)).detach()

    def compute():
        n, nx, ny, nxx, nyy, _, _ = network_derivatives(model.nets[0], z.detach())
        if mixed:
            nt, ntx, nty, ntxx, _, ntxy, ntxxy = network_derivatives(model.nets[0], torch.cat((x, y*0+1), 1))
            d, dx, dxx = n-nt-nty, nx-ntx-ntxy, nxx-ntxx-ntxxy
            u = b+C*d
            uy = by+Cy*d+C*ny
            lap = lapb+Cl*d+2*Cx*dx+C*dxx+2*Cy*ny+C*nyy
            return lap+(u*uy if c == '8' else 0)-rhs
        return lapb+Cl*n+2*Cx*nx+2*Cy*ny+C*(nxx+nyy)-rhs
    return compute

"""Independent implementation of arXiv:physics/9705023v1, Problems 1-8."""
import math
import torch
from torch import nn

torch.set_default_dtype(torch.float64)
CASES = ['1', '2', '3_ivp', '3_bvp', '4', '5', '6', '7', '8']


def derivative(u, z, axis=0):
    return torch.autograd.grad(u.sum(), z, create_graph=True)[0][:, axis:axis+1]


def exact(case, z):
    x = z[:, :1]
    if case == '1':
        return torch.exp(-x*x/2)/(1+x+x**3)+x*x
    if case in ('2', '3_ivp', '3_bvp'):
        return torch.exp(-x/5)*torch.sin(x)
    if case == '4':
        return torch.cat((torch.sin(x), 1+x*x), 1)
    y = z[:, 1:2]
    if case == '5':
        return torch.exp(-x)*(x+y**3)
    if case == '6':
        return torch.exp(-(3*x+y)/5)*torch.sin(9*x*x+y)
    return y*y*torch.sin(math.pi*x)


def forcing(case, z):
    x, y = z[:, :1], z[:, 1:2]
    if case == '5':
        return torch.exp(-x)*(x-2+y**3+6*y)
    if case == '6':
        return torch.exp(-(3*x+y)/5)*((-21.6*x+17.6)*torch.cos(9*x*x+y)
               +(-0.6-324*x*x)*torch.sin(9*x*x+y))
    s = torch.sin(math.pi*x)
    return s*(2-math.pi**2*y*y+(2*y**3*s if case == '8' else 0))


class Trial(nn.Module):
    def __init__(self, case):
        super().__init__()
        self.case = case
        dim = 2 if case in ('5', '6', '7', '8') else 1
        self.nets = nn.ModuleList([nn.Sequential(nn.Linear(dim, 10), nn.Sigmoid(),
                                  nn.Linear(10, 1, bias=False))
                                  for _ in range(2 if case == '4' else 1)])

    def forward(self, z):
        x = z[:, :1]
        n = self.nets[0](z)
        c = self.case
        if c == '1': return 1+x*n
        if c == '2': return x*n
        if c == '3_ivp': return x+x*x*n
        if c == '3_bvp': return x*math.sin(1)*math.exp(-0.2)+x*(1-x)*n
        if c == '4': return torch.cat((x*n, 1+x*self.nets[1](z)), 1)
        y = z[:, 1:2]
        if c in ('5', '6'):
            # Transfinite boundary interpolation, equation (18), not the exact interior.
            f0 = exact(c, torch.cat((x*0, y), 1))
            f1 = exact(c, torch.cat((x*0+1, y), 1))
            g0 = exact(c, torch.cat((x, y*0), 1))
            g1 = exact(c, torch.cat((x, y*0+1), 1))
            p00 = exact(c, z*0)
            p10 = exact(c, torch.cat((x*0+1, y*0), 1))
            p01 = exact(c, torch.cat((x*0, y*0+1), 1))
            p11 = exact(c, z*0+1)
            a = (1-x)*f0+x*f1+(1-y)*(g0-(1-x)*p00-x*p10)+y*(g1-(1-x)*p01-x*p11)
            return a+x*(1-x)*y*(1-y)*n
        top = torch.cat((x, y*0+1), 1)
        nt = self.nets[0](top)
        nt_y = derivative(nt, top, 1)
        return 2*y*torch.sin(math.pi*x)+x*(1-x)*y*(n-nt-nt_y)


def residual(case, z, u):
    x = z[:, :1]
    ux = derivative(u[:, :1], z)
    if case == '1':
        q = (1+3*x*x)/(1+x+x**3)
        return ux+(x+q)*u-(x**3+2*x+x*x*q)
    if case == '2': return ux+u/5-torch.exp(-x/5)*torch.cos(x)
    if case.startswith('3'):
        return derivative(ux, z)+ux/5+u+torch.exp(-x/5)*torch.cos(x)/5
    if case == '4':
        a, b = u[:, :1], u[:, 1:2]
        r1 = ux-(torch.cos(x)+a*a+b-(1+x*x+torch.sin(x)**2))
        r2 = derivative(b, z)-(2*x-(1+x*x)*torch.sin(x)+a*b)
        return torch.cat((r1, r2), 1)
    uy = derivative(u, z, 1)
    lap = derivative(ux, z)+derivative(uy, z, 1)
    return lap+(u*uy if case == '8' else 0)-forcing(case, z)


def domain(case):
    return 3.0 if case == '4' else 2.0 if case in ('2', '3_ivp') else 1.0


def grid(case, n=10, offset=False):
    t = (torch.arange(n)+0.5)/n if offset else torch.linspace(0, 1, n)
    if case in ('5', '6', '7', '8'):
        return torch.cartesian_prod(t, t).requires_grad_(True)
    return (t[:, None]*domain(case)).requires_grad_(True)

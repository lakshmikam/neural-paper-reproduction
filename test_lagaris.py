import unittest
import torch
from lagaris import CASES, Trial, grid, exact, residual, derivative
from fast_residual import make_residual


class Equations(unittest.TestCase):
    def test_fast_pde_residual_and_parameter_gradients(self):
        for c in ('5', '6', '7', '8'):
            m = Trial(c)
            z = grid(c, 7, offset=True)
            a = residual(c, z, m(z))
            b = make_residual(m, z)()
            torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
            ga = torch.autograd.grad(a.square().sum(), tuple(m.parameters()))
            gb = torch.autograd.grad(b.square().sum(), tuple(m.parameters()))
            for x, y in zip(ga, gb):
                torch.testing.assert_close(x, y, atol=1e-10, rtol=1e-10)

    def test_exact_solutions_satisfy_all_equations(self):
        for c in CASES:
            z = grid(c, 17, offset=True)
            self.assertLess(float(residual(c, z, exact(c, z)).abs().max().detach()), 1e-11, c)

    def test_boundary_conditions_hold_for_random_weights(self):
        for c in CASES:
            model = Trial(c)
            if int(c[0]) < 5:
                z = torch.zeros(1, 1, requires_grad=True)
                torch.testing.assert_close(model(z), exact(c, z), atol=1e-12, rtol=0)
                if c == '3_ivp': self.assertAlmostEqual(float(derivative(model(z), z).detach()), 1)
                if c == '3_bvp':
                    z = torch.ones(1, 1, requires_grad=True)
                    torch.testing.assert_close(model(z), exact(c, z))
                continue
            t = torch.linspace(0, 1, 17)[:, None]
            edges = [torch.cat((0*t, t), 1), torch.cat((0*t+1, t), 1), torch.cat((t, 0*t), 1)]
            if c in ('5', '6'): edges.append(torch.cat((t, 0*t+1), 1))
            for z in edges:
                z.requires_grad_(True)
                torch.testing.assert_close(model(z), exact(c, z), atol=1e-12, rtol=0)
            if c in ('7', '8'):
                z = torch.cat((t, 0*t+1), 1).requires_grad_(True)
                torch.testing.assert_close(derivative(model(z), z, 1), derivative(exact(c, z), z, 1), atol=1e-12, rtol=0)

    def test_parameter_gradients_match_finite_difference(self):
        for c in CASES:
            torch.manual_seed(4)
            m = Trial(c)
            z = grid(c, 3, offset=True)
            loss = residual(c, z, m(z)).square().sum()
            loss.backward()
            p = next(m.parameters())
            analytic = float(p.grad.flatten()[0])
            with torch.no_grad(): p.flatten()[0] += 1e-5
            plus = float(residual(c, z, m(z)).square().sum().detach())
            with torch.no_grad(): p.flatten()[0] -= 2e-5
            minus = float(residual(c, z, m(z)).square().sum().detach())
            self.assertAlmostEqual(analytic, (plus-minus)/2e-5, delta=1e-5*max(1, abs(analytic)), msg=c)


if __name__ == '__main__': unittest.main()

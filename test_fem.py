import unittest
import numpy as np
from fem import shape, solve


class FiniteElements(unittest.TestCase):
    def test_q2_basis_reproduces_biquadratics(self):
        nodes = np.array([(x, y) for x in (0, .5, 1) for y in (0, .5, 1)])
        for x, y in ((.19, .71), (.82, .34)):
            p, dx, dy = shape(x, y)
            for a in range(3):
                for b in range(3):
                    nodal = nodes[:, 0]**a*nodes[:, 1]**b
                    self.assertAlmostEqual(p@nodal, x**a*y**b)
                    self.assertAlmostEqual(dx@nodal, a*x**(a-1)*y**b if a else 0)
                    self.assertAlmostEqual(dy@nodal, b*x**a*y**(b-1) if b else 0)

    def test_mesh_refinement_reduces_error(self):
        for c in ('5', '6', '7', '8'):
            coarse = solve(c, 4)[0]
            fine = solve(c, 8)[0]
            self.assertLess(fine['interpolation_max_error'], coarse['interpolation_max_error']/3, c)
            self.assertLess(fine['residual_norms'][-1], 1e-9)


if __name__ == '__main__': unittest.main()

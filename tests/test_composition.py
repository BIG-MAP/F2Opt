import unittest

import torch

from src import composition


class TestCompositionBasicOperators(unittest.TestCase):

    def test_closure(self):
        x = torch.tensor([0.1, 0.2, 0.3, 0.4])
        self.assertTrue(torch.allclose(composition.closure(x), x))
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        self.assertTrue(torch.allclose(composition.closure(x, 10.0), x))

    def test_perturb(self):
        x = torch.tensor([0.1, 0.2, 0.3, 0.4])
        y = torch.tensor([1./6, 1./3, 1./6, 1./3])
        z = torch.tensor([0.0625, 0.25, 0.1875, 0.5])
        self.assertTrue(torch.allclose(composition.perturb(x, y), z))
        self.assertTrue(torch.allclose(composition.perturb(x, torch.ones_like(x)), x))

    def test_perturb_inv(self):
        x = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
        y = torch.tensor([[1./6, 1./3, 1./6, 1./3]])
        z = torch.tensor([0.14285714, 0.14285714, 0.42857143, 0.28571429])
        n = composition.closure(torch.ones_like(x))
        self.assertTrue(torch.allclose(composition.perturb_inv(x, y), z))
        self.assertTrue(torch.allclose(composition.perturb_inv(composition.perturb(x, y), y), x))
        self.assertTrue(torch.allclose(composition.perturb_inv(composition.perturb(x, y), x), y))
        self.assertTrue(torch.allclose(composition.perturb_inv(x, x), n))

    def test_power(self):
        x = torch.tensor([.1, .3, .4, .2])
        y = torch.tensor([0.23059566, 0.25737316, 0.26488486, 0.24714631])
        n = composition.closure(torch.ones_like(x))
        self.assertTrue(torch.allclose(composition.power(x, 0.1), y))
        self.assertTrue(torch.allclose(composition.power(x, 1.0), x))
        self.assertTrue(torch.allclose(composition.power(x, 0.0), n))


class TestCompositionAitchisonGeometry(unittest.TestCase):

    def test_inner(self):
        x = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
        y = torch.tensor([[0.1, 0.1, 0.4, 0.4]])
        z = torch.tensor([[0.4, 0.4, 0.1, 0.1]])
        n = composition.closure(torch.ones_like(x))
        # The neutral composition is the zero vector
        self.assertEqual(composition.inner(n, n).item(), 0.0)
        self.assertEqual(composition.inner(x, n).item(), 0.0)
        # Points in the same direction
        self.assertGreater(composition.inner(x, y).item(), 0.0)
        # Points in opposite directions
        self.assertLess(composition.inner(x, z).item(), 0.0)

    def test_norm(self):
        x = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
        y = torch.tensor([[0.1, 0.7, 0.1, 0.1]])
        n = composition.closure(torch.ones_like(x))
        self.assertGreater(composition.norm(x).item(), 0.0)
        self.assertEqual(composition.norm(n).item(), 0.0)
        self.assertGreater(composition.norm(y).item(), composition.norm(x).item())

    def test_dist(self):
        x = torch.tensor([
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.3, 0.3, 0.3],
        ])
        y = torch.tensor([
            [0.1, 0.3, 0.3, 0.3],
            [0.1, 0.2, 0.3, 0.4],
        ])
        d = composition.dist(x, x)
        self.assertTrue(torch.allclose(d, torch.zeros_like(d)))
        d = composition.dist(x, y)
        self.assertTrue(torch.all(d == d[0]))


class TestCompositionTransformations(unittest.TestCase):

    def test_clr(self):
        x = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
        y = torch.tensor([[0.1, 0.3, 0.3, 0.3]])
        n = composition.closure(torch.ones_like(x))
        c = 0.987654321
        self.assertAlmostEqual(composition.clr(x).sum().item(), 0.0, 5)
        self.assertTrue(torch.allclose(composition.clr(n), torch.zeros_like(n)))
        self.assertTrue(torch.allclose(composition.clr_inv(composition.clr(x)), x))
        self.assertTrue(torch.allclose(composition.clr_inv(composition.clr(y)), y))
        self.assertTrue(composition.clr(x).shape[-1] == x.shape[-1])
        # linearity
        self.assertTrue(torch.allclose(
            composition.clr(composition.perturb(x, y)),
            composition.clr(x) + composition.clr(y)
        ))
        self.assertTrue(torch.allclose(
            composition.clr(composition.power(x, c)),
            composition.clr(x) * c
        ))
        # inner product
        self.assertTrue(torch.allclose(
            composition.inner(x, y),
            composition.clr(x) @ composition.clr(y).T
        ))
        self.assertTrue(torch.allclose(
            composition.norm(x),
            torch.linalg.norm(composition.clr(x))
        ))
        # distance
        self.assertTrue(torch.allclose(
            composition.dist(x, y),
            torch.linalg.norm(composition.clr(x) - composition.clr(y), dim=1)
        ))

    def test_ilr(self):
        x = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
        y = torch.tensor([[0.1, 0.3, 0.3, 0.3]])
        # n = composition.closure(torch.ones_like(x))
        c = 0.987654321
        # self.assertAlmostEqual(composition.clr(x).sum().item(), 0.0, 5)
        # self.assertTrue(torch.allclose(composition.clr(n), torch.zeros_like(n)))
        self.skipTest("TODO: Finish implementation.")
        self.assertTrue(torch.allclose(composition.ilr_inv(composition.ilr(x)), x))
        self.assertTrue(torch.allclose(composition.ilr_inv(composition.ilr(y)), y))
        self.assertTrue(composition.ilr(x).shape[-1] == x.shape[-1] - 1)
        # linearity
        self.assertTrue(torch.allclose(
            composition.ilr(composition.perturb(x, y)),
            composition.ilr(x) + composition.ilr(y)
        ))
        self.assertTrue(torch.allclose(
            composition.ilr(composition.power(x, c)),
            composition.ilr(x) * c
        ))
        # inner product
        self.assertTrue(torch.allclose(
            composition.inner(x, y),
            composition.ilr(x) @ composition.ilr(y).T
        ))
        self.assertTrue(torch.allclose(
            composition.norm(x),
            torch.linalg.norm(composition.ilr(x))
        ))
        # distance
        self.assertTrue(torch.allclose(
            composition.dist(x, y),
            torch.linalg.norm(composition.ilr(x) - composition.ilr(y), dim=1)
        ))


class TestCompositionSampling(unittest.TestCase):

    def test_sample_compositions(self):
        """Test sample compositions."""
        num_features, num_samples = 10, 1000
        x = composition.sample_compositions(num_features, num_samples)
        self.assertEqual(x.shape, (num_samples, num_features))
        self.assertTrue((x >= 0).all())
        self.assertTrue((x <= 1).all())
        self.assertTrue((x.sum(dim=1) == 1).all())
        self.assertTrue((x.min(0)[0] < 0.05).all())

    def test_sample_compositions_single(self):
        """Test compositions with single feature."""
        num_features, num_samples = 1, 10
        x = composition.sample_compositions(num_features, num_samples)
        self.assertEqual(x.shape, (num_samples, num_features))
        self.assertTrue((x == 1.0).all())

    def test_sample_compositions_with_constraints_single(self):
        """Test compositions with single feature."""
        num_features, num_samples = 1, 10
        lower = torch.tensor([0.1])
        upper = torch.tensor([1.0])
        tolerance = torch.tensor([0.01])
        x = composition.sample_compositions_with_constraints(lower, upper, tolerance, num_samples)
        self.assertEqual(x.shape, (num_samples, num_features))
        self.assertTrue((x == 1.0).all())

    def test_sample_compositions_with_constraints_negative(self):
        """Test compositions are not sampled outside the given constraints."""
        num_features, num_samples = 10, 1000
        # Constraints
        tolerance = 0.01
        lower = torch.rand(num_features) * (0.5 / num_features)
        lower[lower < tolerance] = 0
        lower[0] = 0.1
        upper = torch.ones(num_features) - torch.rand(num_features) * (0.5 / num_features)
        upper[1] = 0.5
        tolerance = torch.ones(num_features) * tolerance
        # Sample compositions
        x = composition.sample_compositions_with_constraints(lower, upper, tolerance, num_samples)
        # Check
        self.assertEqual(x.shape, (num_samples, num_features))
        self.assertTrue((x >= 0).all())
        self.assertTrue((x <= 1).all())
        self.assertTrue((torch.abs(x.sum(dim=1) - 1.0) < 1e-6).all())
        self.assertTrue(((x == 0) | (x >= tolerance)).all())
        self.assertTrue((x >= lower).all())
        self.assertTrue((x <= upper).all())

    def test_sample_compositions_with_constraints_positive(self):
        """Test the sampled compositions cover the volume inside the given constraints."""
        num_features, num_samples = 2, 1000
        # Sample without constraints and check that min and max values are close to 0 and 1
        zeros = torch.zeros(num_features)
        ones = torch.ones(num_features)
        x = composition.sample_compositions_with_constraints(zeros, ones, zeros, num_samples)
        self.assertEqual(x.shape, (num_samples, num_features))
        self.assertTrue((torch.abs(x.sum(dim=1) - 1.0) < 1e-6).all())
        self.assertTrue((x.min(0)[0] < 0.05).all())
        self.assertTrue((x.max(0)[0] > 0.95).all())
        # Sample with lower and upper bounds and check that values are close to the bounds
        lower = torch.tensor([0.05, 0.05])
        upper = torch.tensor([0.95, 0.95])
        x = composition.sample_compositions_with_constraints(lower, upper, zeros, num_samples)
        self.assertEqual(x.shape, (num_samples, num_features))
        self.assertTrue((torch.abs(x.sum(dim=1) - 1.0) < 1e-6).all())
        self.assertTrue((x.min(0)[0] < (lower + 0.05)).all())
        self.assertTrue((x.max(0)[0] > (upper - 0.05)).all())
        # Sample with tolerance and check that values are rounded to exactly 0 and 1
        x = composition.sample_compositions_with_constraints(zeros, ones, lower, num_samples)
        self.assertEqual(x.shape, (num_samples, num_features))
        self.assertTrue((torch.abs(x.sum(dim=1) - 1.0) < 1e-6).all())
        self.assertTrue((x.min(0)[0] == 0.0).all())
        self.assertTrue((x.max(0)[0] == 1.0).all())

import torch
import unittest
import numpy as np
import warp_rnnt._C as warp_rnnt_core


xs = torch.tensor([], dtype=torch.float32)
ys = torch.tensor([], dtype=torch.int)
xn = torch.tensor([], dtype=torch.int)
yn = torch.tensor([], dtype=torch.int)


class WRNNTLossTest(unittest.TestCase):

    def test_contiguous(self):
        xs = torch.tensor(np.zeros((4, 3, 2, 1)), dtype=torch.float32).transpose(0, 1)
        with self.assertRaisesRegex(RuntimeError, "xs must be contiguous"):
            warp_rnnt_core.rnnt_loss(xs, ys, xn, yn)

    def test_device(self):
        with self.assertRaisesRegex(RuntimeError, "xs must be located in the CUDA"):
            warp_rnnt_core.rnnt_loss(xs, ys, xn, yn)

    def test_shape(self):
        with self.assertRaisesRegex(RuntimeError, "xs must have 4 dimensions"):
            warp_rnnt_core.rnnt_loss(xs.cuda(), ys.cuda(), xn.cuda(), yn.cuda())

    def test_type(self):
        ys = torch.tensor([], dtype=torch.long)
        with self.assertRaisesRegex(RuntimeError, "ys must be a Int tensor"):
            warp_rnnt_core.rnnt_loss(xs, ys, xn, yn)

    def test_forward_single(self):

        xs = torch.tensor(
            [[[[0.1, 0.6, 0.1, 0.1, 0.1],
               [0.1, 0.1, 0.6, 0.1, 0.1],
               [0.1, 0.1, 0.2, 0.8, 0.1]],
              [[0.1, 0.6, 0.1, 0.1, 0.1],
               [0.1, 0.1, 0.2, 0.1, 0.1],
               [0.7, 0.1, 0.2, 0.1, 0.1]]]],
            dtype=torch.float32)

        xs = torch.nn.functional.log_softmax(xs, dim=-1)
        ys = torch.tensor([[1, 2]], dtype=torch.int)

        xn = torch.tensor([2], dtype=torch.int)
        yn = torch.tensor([2], dtype=torch.int)

        costs, grads = warp_rnnt_core.rnnt_loss(
            xs.cuda(), ys.cuda(),
            xn.cuda(), yn.cuda())

        expected_cost = 4.495666

        np.testing.assert_almost_equal(costs.item(), expected_cost, decimal=6)

        expected_grads = np.array(
            [[[[-0.308198071906, -0.6918019280939998, 0.0, 0.0, 0.0],
               [-0.308198071906, 0.0, -0.3836038561880001, 0.0, 0.0],
               [-0.3836038561880001, 0.0, 0.0, 0.0, 0.0]],
              [[0.0, -0.308198071906, 0.0, 0.0, 0.0],
               [0.0, 0.0, -0.6163961438119995, 0.0, 0.0],
               [-0.9999999999999991, 0.0, 0.0, 0.0, 0.0]]]])

        np.testing.assert_array_almost_equal(grads.cpu().numpy(), expected_grads)

    def test_forward_batch(self):

        xs = torch.tensor(
            [
                [[[0.1, 0.6, 0.1, 0.1, 0.1],
                  [0.1, 0.1, 0.6, 0.1, 0.1],
                  [0.1, 0.1, 0.2, 0.8, 0.1]],
                 [[0.1, 0.6, 0.1, 0.1, 0.1],
                  [0.1, 0.1, 0.2, 0.1, 0.1],
                  [0.7, 0.1, 0.2, 0.1, 0.1]],
                 [[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]]],

                [[[0.1, 0.6, 0.1, 0.1, 0.1],
                  [0.1, 0.1, 0.6, 0.1, 0.1],
                  [0.1, 0.1, 0.2, 0.8, 0.1]],
                 [[0.1, 0.6, 0.1, 0.1, 0.1],
                  [0.1, 0.1, 0.2, 0.1, 0.1],
                  [0.7, 0.1, 0.2, 0.1, 0.1]],
                 [[0.1, 0.6, 0.1, 0.1, 0.1],
                  [0.1, 0.1, 0.6, 0.1, 0.1],
                  [0.1, 0.1, 0.2, 0.8, 0.1]]]
            ],
            dtype=torch.float32)

        xs = torch.nn.functional.log_softmax(xs, dim=-1)
        ys = torch.tensor([[1, 2], [1, 2]], dtype=torch.int)

        xn = torch.tensor([2, 3], dtype=torch.int)
        yn = torch.tensor([2, 2], dtype=torch.int)

        costs, grads = warp_rnnt_core.rnnt_loss(
            xs.cuda(), ys.cuda(),
            xn.cuda(), yn.cuda())

        expected_costs = np.array([4.495666773770733, 5.7367250428101615])

        np.testing.assert_array_almost_equal(costs.cpu().numpy(), expected_costs, decimal=6)

        expected_grads = np.array([

            [[[-0.308198071906, -0.6918019280939998, 0.0, 0.0, 0.0],
              [-0.308198071906, 0.0, -0.3836038561880001, 0.0, 0.0],
              [-0.3836038561880001, 0.0, 0.0, 0.0, 0.0]],
             [[0.0, -0.308198071906, 0.0, 0.0, 0.0],
              [0.0, 0.0, -0.6163961438119995, 0.0, 0.0],
              [-0.9999999999999991, 0.0, 0.0, 0.0, 0.0]],
             [[0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0]]],

            [[[-0.45920877, -0.54079123, -0.,         -0.,         -0.],
              [-0.32392462, -0.,         -0.21686661, -0.,         -0.],
              [-0.21686661, -0.,         -0.,         -0.,         -0.]],
             [[-0.13528414, -0.32392462, -0.,         -0.,         -0.],
              [-0.29937584, -0.,         -0.3484734,  -0.,         -0.],
              [-0.56534001, -0.,         -0.,         -0.,         -0.]],
             [[-0.,         -0.13528414, -0.,         -0.,         -0.],
              [-0.,         -0.,         -0.43465999, -0.,         -0.],
              [-1.,         -0.,         -0.,         -0.,         -0.]]]
        ])

        np.testing.assert_array_almost_equal(grads.cpu().numpy(), expected_grads)


if __name__ == "__main__":
    unittest.main()

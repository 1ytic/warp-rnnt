import os
import tensorflow as tf
import numpy as np
from transducer_tensorflow import transducer_loss
from scipy.special import log_softmax


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

np.random.seed(42)
tf.set_random_seed(42)


class RNNTLossTest(tf.test.TestCase):

    def _run_transducer(self, xs, xn,
                        ys, yn,
                        expected_costs, expected_grads,
                        blank=0,
                        use_gpu=True, expected_error=None,
                        gather=False):

        with tf.device("/gpu:0"):
            xs_t = tf.constant(xs)
            xn_t = tf.constant(xn)
            ys_t = tf.constant(ys)
            yn_t = tf.constant(yn)
            costs = transducer_loss(
                log_probs=xs_t,
                labels=ys_t,
                frames_lengths=xn_t,
                labels_lengths=yn_t,
                average_frames=False,
                reduction=None,
                blank=blank,
                gather=gather)

            grad = tf.gradients(costs, [xs_t])[0]

        log_dev_placement = False
        if not use_gpu:
            # Note: using use_gpu=False seems to not work
            # it runs the GPU version instead
            config = tf.ConfigProto(log_device_placement=log_dev_placement,
                                    device_count={'GPU': 0})
        else:
            config = tf.ConfigProto(log_device_placement=log_dev_placement,
                                    allow_soft_placement=False)

        with self.test_session(use_gpu=use_gpu, force_gpu=use_gpu, config=config) as sess:
            if expected_error is None:
                (tf_costs, tf_grad) = sess.run([costs, grad])
                if not isinstance(expected_costs, type(None)):
                    self.assertAllClose(tf_costs, expected_costs, atol=1e-6)
                if not isinstance(expected_grads, type(None)):
                    self.assertAllClose(tf_grad, expected_grads, atol=1e-6)
            else:
                with self.assertRaisesOpError(expected_error):
                    sess.run([costs, grad])

                    sess.run([costs, grad])

    def test_one_to_many(self):
        xs = np.asarray(
            [[[[0.1, 0.6, 0.1, 0.1, 0.1],
               [0.1, 0.1, 0.6, 0.1, 0.1],
               [0.1, 0.1, 0.2, 0.8, 0.1]]]],
            dtype=np.float32)
        xs = log_softmax(xs, axis=-1)
        ys = np.asarray([[1, 2]], dtype=np.int32)
        xn = np.asarray([1], dtype=np.int32)
        yn = np.asarray([2], dtype=np.int32)
        expected_costs = np.asarray([4.274244594423859], dtype=np.float32)
        expected_grads = np.asarray(
            [[[[0.0, -1., 0.0, 0.0, 0.0],
               [0.0, 0.0, -1., 0.0, 0.0],
               [-1., 0.0, 0.0, 0.0, 0.0]]]],
            dtype=np.float32)

        self._run_transducer(xs, xn,
                             ys, yn,
                             expected_costs, expected_grads,
                             use_gpu=True, expected_error=None)

    def test_one_to_empty(self):
        xs = np.asarray(
            [[[[0.1, 0.6, 0.1, 0.1, 0.1]]]], dtype=np.float32)
        xs = log_softmax(xs, axis=-1)
        ys = np.asarray([[]], dtype=np.int32)
        xn = np.asarray([1], dtype=np.int32)
        yn = np.asarray([0], dtype=np.int32)

        expected_costs = np.asarray([1.7314291957733714], dtype=np.float32)
        expected_grads = np.asarray(
            [[[[-1., 0.0, 0.0, 0.0, 0.0]]]],
            dtype=np.float32)

        self._run_transducer(xs, xn,
                             ys, yn,
                             expected_costs, expected_grads,
                             use_gpu=True, expected_error=None)

    def test_forward_single(self):
        xs = np.asarray(
            [[[[0.1, 0.6, 0.1, 0.1, 0.1],
               [0.1, 0.1, 0.6, 0.1, 0.1],
               [0.1, 0.1, 0.2, 0.8, 0.1]],
              [[0.1, 0.6, 0.1, 0.1, 0.1],
               [0.1, 0.1, 0.2, 0.1, 0.1],
               [0.7, 0.1, 0.2, 0.1, 0.1]]]],
            dtype=np.float32)
        xs = log_softmax(xs, axis=-1)
        ys = np.asarray([[1, 2]], dtype=np.int32)
        xn = np.asarray([2], dtype=np.int32)
        yn = np.asarray([2], dtype=np.int32)
        expected_cost = 4.495666
        expected_costs = np.asarray([expected_cost], dtype=np.float32)
        expected_grads = np.array(
            [[[[-0.308198071906, -0.6918019280939998, 0.0, 0.0, 0.0],
               [-0.308198071906, 0.0, -0.3836038561880001, 0.0, 0.0],
               [-0.3836038561880001, 0.0, 0.0, 0.0, 0.0]],
              [[0.0, -0.308198071906, 0.0, 0.0, 0.0],
               [0.0, 0.0, -0.6163961438119995, 0.0, 0.0],
               [-0.9999999999999991, 0.0, 0.0, 0.0, 0.0]]]],
            dtype=np.float32)

        self._run_transducer(xs, xn,
                             ys, yn,
                             expected_costs, expected_grads,
                             use_gpu=True, expected_error=None)

    def test_forward_batch(self):

        xs = np.asarray(
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
            dtype=np.float32)
        xs = log_softmax(xs, axis=-1)

        ys = np.asarray([[1, 2], [1, 2]], dtype=np.int32)

        xn = np.asarray([2, 3], dtype=np.int32)
        yn = np.asarray([2, 2], dtype=np.int32)

        expected_costs = np.array(
            [4.495666773770733, 5.7367250428101615], dtype=np.float32)

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
        ],
            dtype=np.float32)

        self._run_transducer(xs, xn,
                             ys, yn,
                             expected_costs, expected_grads,
                             use_gpu=True, expected_error=None)

    def test_calls(self):

        n = 128
        t = 100
        u = 90
        v = 3

        for i in range(2):

            rng = np.random.RandomState(i)

            xs = rng.randn(n, t, u, v)
            xs = np.asarray(xs, dtype=np.float32)
            xs = log_softmax(xs, axis=-1)

            ys = np.asarray(
                rng.randint(1, v, (n, u-1)), dtype=np.int32)

            xn = np.asarray([t] * n, dtype=np.int32)
            yn = np.asarray(rng.randint(1, u, n), dtype=np.int32)

            # costs, grads = transducer_loss(
            #     xs, ys,
            #     xn, yn)
            self._run_transducer(xs, xn,
                                 ys, yn,
                                 expected_costs=None, expected_grads=None,
                                 use_gpu=True, expected_error=None)

    def test_forward_single_gather(self, blank=0):

        xs = np.asarray(
            [[[[0.1, 0.6, 0.1, 0.1, 0.1],
               [0.1, 0.1, 0.6, 0.1, 0.1],
               [0.1, 0.1, 0.2, 0.8, 0.1]],
              [[0.1, 0.6, 0.1, 0.1, 0.1],
               [0.1, 0.1, 0.2, 0.1, 0.1],
               [0.7, 0.1, 0.2, 0.1, 0.1]]]],
            dtype=np.float32)
        xs = log_softmax(xs, axis=-1)

        ys = np.asarray([[1, 2]], dtype=np.int32)

        xn = np.asarray([2], dtype=np.int32)
        yn = np.asarray([2], dtype=np.int32)

        N, T, U, V = xs.shape
        index = np.full([N, T, U, 2], np.array(blank, dtype=np.int64))
        index[:, :, :U-1, 1] = np.expand_dims(ys, axis=1)
        xs = np.take_along_axis(xs, indices=index, axis=3)

        expected_costs = np.array(
            [4.495666], dtype=np.float32)

        expected_grads = np.array(
            [[[[-0.308198071906, -0.6918019280939998],
               [-0.308198071906, -0.3836038561880001],
               [-0.3836038561880001, 0.0]],
              [[0.0, -0.308198071906],
               [0.0, -0.6163961438119995],
               [-0.9999999999999991, 0.0]]]])

        self._run_transducer(xs, xn,
                             ys, yn,
                             expected_costs=expected_costs, expected_grads=expected_grads,
                             use_gpu=True, expected_error=None, blank=-1)

    def test_forward_single_inner_gather(self, blank=0):
        xs = np.asarray(
            [[[[0.1, 0.6, 0.1, 0.1, 0.1],
               [0.1, 0.1, 0.6, 0.1, 0.1],
               [0.1, 0.1, 0.2, 0.8, 0.1]],
              [[0.1, 0.6, 0.1, 0.1, 0.1],
               [0.1, 0.1, 0.2, 0.1, 0.1],
               [0.7, 0.1, 0.2, 0.1, 0.1]]]],
            dtype=np.float32)
        xs = log_softmax(xs, axis=-1)
        ys = np.asarray([[1, 2]], dtype=np.int32)
        xn = np.asarray([2], dtype=np.int32)
        yn = np.asarray([2], dtype=np.int32)
        expected_cost = 4.495666
        expected_costs = np.asarray([expected_cost], dtype=np.float32)
        expected_grads = np.array(
            [[[[-0.308198071906, -0.6918019280939998, 0.0, 0.0, 0.0],
               [-0.308198071906, 0.0, -0.3836038561880001, 0.0, 0.0],
               [-0.3836038561880001, 0.0, 0.0, 0.0, 0.0]],
              [[0.0, -0.308198071906, 0.0, 0.0, 0.0],
               [0.0, 0.0, -0.6163961438119995, 0.0, 0.0],
               [-0.9999999999999991, 0.0, 0.0, 0.0, 0.0]]]],
            dtype=np.float32)

        self._run_transducer(xs, xn,
                             ys, yn,
                             expected_costs=expected_costs, expected_grads=expected_grads,
                             use_gpu=True, expected_error=None, gather=True)


if __name__ == "__main__":
    tf.test.main()

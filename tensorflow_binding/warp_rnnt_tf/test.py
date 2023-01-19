import numpy as np
import tensorflow as tf
from scipy.special import log_softmax
from tensorflow.python.framework.ops import disable_eager_execution
from . import rnnt_loss

np.random.seed(42)
tf.random.set_seed(42)


class RNNTLossTest(tf.test.TestCase):

    def _run_transducer(self, xs, xn, ys, yn,
                        expected_costs=None, expected_grads=None,
                        blank=0, gather=False):

        @tf.function
        def graph_execution():
          xs_t = tf.constant(xs)
          xn_t = tf.constant(xn)
          ys_t = tf.constant(ys)
          yn_t = tf.constant(yn)
          costs = rnnt_loss(
              log_probs=xs_t,
              labels=ys_t,
              frames_lengths=xn_t,
              labels_lengths=yn_t,
              average_frames=False,
              reduction=None,
              blank=blank,
              gather=gather)
          grads = tf.gradients(costs, xs_t)[0]
          return costs, grads

        with tf.device("/device:GPU:0"):
          costs1, grad1 = graph_execution()

        with self.session(use_gpu=True, force_gpu=True) as sess:
            tf_costs, tf_grad = sess.run([costs1, grad1])
            if expected_costs is not None:
                self.assertAllClose(tf_costs, expected_costs, atol=1e-6)
            if expected_grads is not None:
                self.assertAllClose(tf_grad, expected_grads, atol=1e-6)

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

        self._run_transducer(xs, xn, ys, yn,
                             expected_costs, expected_grads)

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

        self._run_transducer(xs, xn, ys, yn,
                             expected_costs, expected_grads)

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
        expected_costs = np.asarray([4.495666], dtype=np.float32)
        expected_grads = np.array(
            [[[[-0.308198071906, -0.6918019280939998, 0.0, 0.0, 0.0],
               [-0.308198071906, 0.0, -0.3836038561880001, 0.0, 0.0],
               [-0.3836038561880001, 0.0, 0.0, 0.0, 0.0]],
              [[0.0, -0.308198071906, 0.0, 0.0, 0.0],
               [0.0, 0.0, -0.6163961438119995, 0.0, 0.0],
               [-0.9999999999999991, 0.0, 0.0, 0.0, 0.0]]]],
            dtype=np.float32)

        self._run_transducer(xs, xn, ys, yn,
                             expected_costs, expected_grads)

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

        self._run_transducer(xs, xn, ys, yn,
                             expected_costs, expected_grads)

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

            self._run_transducer(xs, xn, ys, yn)

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

        expected_costs = np.array([4.495666], dtype=np.float32)
        expected_grads = np.array(
            [[[[-0.308198071906, -0.6918019280939998],
               [-0.308198071906, -0.3836038561880001],
               [-0.3836038561880001, 0.0]],
              [[0.0, -0.308198071906],
               [0.0, -0.6163961438119995],
               [-0.9999999999999991, 0.0]]]])

        self._run_transducer(xs, xn, ys, yn,
                             expected_costs, expected_grads,
                             blank=-1)

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
        expected_costs = np.asarray([4.495666], dtype=np.float32)
        expected_grads = np.array(
            [[[[-0.308198071906, -0.6918019280939998, 0.0, 0.0, 0.0],
               [-0.308198071906, 0.0, -0.3836038561880001, 0.0, 0.0],
               [-0.3836038561880001, 0.0, 0.0, 0.0, 0.0]],
              [[0.0, -0.308198071906, 0.0, 0.0, 0.0],
               [0.0, 0.0, -0.6163961438119995, 0.0, 0.0],
               [-0.9999999999999991, 0.0, 0.0, 0.0, 0.0]]]],
            dtype=np.float32)

        self._run_transducer(xs, xn, ys, yn,
                             expected_costs, expected_grads,
                             gather=True)


if __name__ == "__main__":
    disable_eager_execution()
    tf.test.main()

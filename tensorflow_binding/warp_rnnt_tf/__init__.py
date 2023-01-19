import imp
import tensorflow as tf
from tensorflow.python.framework import ops
from typing import Optional, AnyStr

lib_file = imp.find_module('kernels', __path__)[1]
_warp_transducer = tf.load_op_library(lib_file)


def gather_log_probs(log_probs, labels, blank=0):
    """ Gather log_probs with the shape (N, T, U, V) into the shape (N, T, U, 2)
        only for blank and labels values. It's a slightly memory-efficient method.

   Args:
        log_probs (torch.FloatTensor): Input tensor with shape (N, T, U, V)
            where N is the minibatch size, T is the maximum number of
            input frames, U is the maximum number of output labels and V is
            the vocabulary of labels (including the blank).
        labels (torch.IntTensor): Tensor with shape (N, U-1) representing the
            reference labels for all samples in the minibatch.
        blank (int, optional): label used to represent the blank symbol.
            Default: 0.
    """
    def shape_list(x):
        """Deal with dynamic shape in tensorflow cleanly."""
        static = x.shape.as_list()
        dynamic = tf.shape(x)
        return [dynamic[i] if s is None else s for i, s in enumerate(static)]

    def gather(x, indices, gather_axis):
        # if pytorch gather indices are
        # [[[0, 10, 20], [0, 10, 20], [0, 10, 20]],
        #  [[0, 10, 20], [0, 10, 20], [0, 10, 20]]]
        # tf nd_gather needs to be
        # [[0,0,0], [0,0,10], [0,0,20], [0,1,0], [0,1,10], [0,1,20], [0,2,0], [0,2,10], [0,2,20],
        #  [1,0,0], [1,0,10], [1,0,20], [1,1,0], [1,1,10], [1,1,20], [1,2,0], [1,2,10], [1,2,20]]

        N, T, U, V = shape_list(indices)

        # create a tensor containing indices of each element
        all_indices = tf.where(
            tf.fill([N, T, U, V], tf.constant(True, dtype=tf.bool)))
        gather_locations = tf.reshape(indices, [N*T*U*V])

        # splice in our pytorch style index at the correct axis
        gathered = tf.gather_nd(x, tf.stack([
            all_indices[:, 0],
            all_indices[:, 1],
            all_indices[:, 2],
            gather_locations
        ], axis=-1))

        return tf.reshape(gathered, [N, T, U, V])

    # Casts to int64
    blank = tf.cast(blank, dtype=tf.int64)
    labels = tf.cast(labels, dtype=tf.int64)

    # Returns the shape of log_probs
    N, T, U, V = shape_list(log_probs)

    # (N, U-1) -> (N, 1, U-1)
    index = tf.expand_dims(labels, axis=1)
    # (N, 1, U-1) -> (N, T, U-1)
    index = tf.tile(index, tf.stack([1, T, 1]))
    # (N, 1, U-1) -> (N, T, U-1, 1)
    index = tf.expand_dims(index, axis=-1)
    # (N, T, U-1,1)
    blank_index = tf.fill([N, T, U-1, 1], blank)
    # (N, T, U-1,1), (N, T, U-1,1)] -> (N, T, U-1, 2)
    index = tf.concat([blank_index, index], axis=-1)
    # (N, T, 1,2)
    blank_index = tf.fill([N, T, 1, 2], blank)
    # (N, T, U-1, 2), (N, T, 1, 2) -> (N, T, U, 2)
    index = tf.concat([index, blank_index], axis=2)

    return gather(log_probs, index, 3)


def rnnt_loss(
        log_probs, labels, frames_lengths, labels_lengths,
        average_frames: bool = False,
        reduction: Optional[AnyStr] = None,
        blank: int = 0,
        gather: bool = False,
        fastemit_lambda: float = 0.0):
    """The CUDA-Warp Transducer loss.

    Args:
        log_probs (Tensor): Float Tensor with shape (N, T, U, V)
            where N is the minibatch size, T is the maximum number of
            input frames, U is the maximum number of output labels and V is
            the vocabulary of labels (including the blank).
        labels (Tensor): Integer Tensor with shape (N, U-1) representing the
            reference labels for all samples in the minibatch.
        frames_lengths (Tensor): Integer Tensor with shape (N,) representing the
            number of frames for each sample in the minibatch.
        labels_lengths (Tensor): Integer Tensor with shape (N,) representing the
            length of the transcription for each sample in the minibatch.
        average_frames (bool, optional): Specifies whether the loss of each
            sample should be divided by its number of frames.
            Default: False.
        reduction (string, optional): Specifies the type of reduction.
            Default: None.
        blank (int, optional): label used to represent the blank symbol.
            Default: 0.
        gather (bool, optional): Reduce memory consumption.
            Default: False.
        fastemit_lambda (float, optional): FastEmit regularization
            (https://arxiv.org/abs/2010.11148).
            Default: 0.0.
    """
    assert average_frames is None or isinstance(average_frames, bool)
    assert reduction is None or reduction in ("none", "mean", "sum")
    assert isinstance(blank, int)

    if gather:
        log_probs = gather_log_probs(log_probs, labels, blank)
        blank = -1

    costs, _ = _warp_transducer.transducer_loss(
        log_probs, labels, frames_lengths, labels_lengths, blank, fastemit_lambda)

    if average_frames:
        costs = costs / frames_lengths  # (N,)

    if reduction == "sum":
        return tf.reduce_sum(costs)
    elif reduction == "mean":
        return tf.reduce_mean(costs)
    return costs


@ops.RegisterGradient("TransducerLoss")
def _TransducerLossGrad(op, grad_loss, _):
    """The derivative provided by Transducer Loss.

    Args:
       op: the TransducerLoss op.
       grad_loss: The backprop for cost.

    Returns:
       The Transducer Loss gradient.
    """
    grad = op.outputs[1]
    # NOTE since here we are batch first, cannot use _BroadcastMul
    grad_loss = tf.reshape(grad_loss, (-1, 1, 1, 1))
    return [grad_loss * grad, None, None, None]

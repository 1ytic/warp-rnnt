import imp
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops.nn_grad import _BroadcastMul
from typing import Optional, AnyStr

lib_file = imp.find_module('kernels', __path__)[1]
_warp_transducer = tf.load_op_library(lib_file)


def transducer_loss(
        log_probs, labels, frames_lengths, labels_lengths,
        average_frames: bool = False,
        reduction: Optional[AnyStr] = None,
        blank: int = 0):
    """The CUDA-Warp Transducer loss.

    Args:
        log_probs (FloatTensor): Input tensor with shape (N, T, U, V)
            where N is the minibatch size, T is the maximum number of
            input frames, U is the maximum number of output labels and V is
            the vocabulary of labels (including the blank).
        labels (IntTensor): Tensor with shape (N, U-1) representing the
            reference labels for all samples in the minibatch.
        frames_lengths (IntTensor): Tensor with shape (N,) representing the
            number of frames for each sample in the minibatch.
        labels_lengths (IntTensor): Tensor with shape (N,) representing the
            length of the transcription for each sample in the minibatch.
        average_frames (bool, optional): Specifies whether the loss of each
            sample should be divided by its number of frames.
            Default: False.
        reduction (string, optional): Specifies the type of reduction.
            Default: None.
        blank (int, optional): label used to represent the blank symbol.
            Default: 0.
    """
    assert average_frames is None or isinstance(average_frames, bool)
    assert reduction is None or reduction in ("none", "mean", "sum")
    assert isinstance(blank, int)

    costs, _ = _warp_transducer.transducer_loss(
        log_probs, labels, frames_lengths, labels_lengths, blank)

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


@ops.RegisterShape("TransducerLoss")
def _TransducerLossShape(op):
    inputs_shape = op.inputs[0].get_shape().with_rank(4)
    batch_size = inputs_shape[0]
    return [batch_size, inputs_shape]

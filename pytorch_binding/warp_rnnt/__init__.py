import torch
import warp_rnnt._C as core
from typing import Optional, AnyStr, Literal
from pkg_resources import get_distribution

__version__ = get_distribution('warp_rnnt').version


class RNNTLoss(torch.autograd.Function):

    @staticmethod
    def forward(ctx, log_probs, labels, frames_lengths, labels_lengths, blank=0, fastemit_lambda=0.0):
        costs, ctx.grads = core.rnnt_loss(
            xs=log_probs, ys=labels,
            xn=frames_lengths, yn=labels_lengths,
            blank=blank,
            fastemit_lambda=fastemit_lambda,
        )
        return costs

    @staticmethod
    def backward(ctx, grads_output):
        grads_output = grads_output.view(-1, 1, 1, 1).to(ctx.grads)
        return ctx.grads.mul_(grads_output), None, None, None, None, None, None

class RNNTLossCompact(torch.autograd.Function):

    @staticmethod
    def forward(ctx, log_probs, labels, frames_lengths, labels_lengths, blank=0, fastemit_lambda=0.0, enable_grad: bool = True):

        costs, grads, loc = core.rnnt_loss_compact(
            xs=log_probs, ys=labels,
            xn=frames_lengths, yn=labels_lengths,
            blank=blank,
            fastemit_lambda=fastemit_lambda,
            required_grad=enable_grad
        )
        if enable_grad:
            cumlen = torch.cumsum(frames_lengths * (labels_lengths+1), dim=0, dtype=torch.int32)
            ctx.V = log_probs.size(-1)
            ctx.blank = blank
            ctx.save_for_backward(grads, loc, cumlen)
        return costs

    @staticmethod
    def backward(ctx, grads_output):
        grads, loc, cumlen = ctx.saved_tensors
        grads_input = core.rnnt_loss_compact_backward(
            grads_output.contiguous(), 
            grads, cumlen,
            loc, ctx.V, ctx.blank
        )

        return grads_input, None, None, None, None, None, None


def rnnt_loss(log_probs: torch.FloatTensor,
              labels: torch.IntTensor,
              frames_lengths: torch.IntTensor,
              labels_lengths: torch.IntTensor,
              average_frames: bool = False,
              reduction: Literal['sum', 'mean', 'none'] = 'none',
              blank: int = 0,
              gather: bool = False,
              fastemit_lambda: float = 0.0,
              compact: bool = False) -> torch.Tensor:

    """The CUDA-Warp RNN-Transducer loss.

    Args:
        log_probs (torch.FloatTensor): Input tensor with shape (N, T, U, V)
            where N is the minibatch size, T is the maximum number of
            input frames, U is the maximum number of output labels and V is
            the vocabulary of labels (including the blank).
        labels (torch.IntTensor): Tensor with shape (N, U-1) representing the
            reference labels for all samples in the minibatch.
        frames_lengths (torch.IntTensor): Tensor with shape (N,) representing the
            number of frames for each sample in the minibatch.
        labels_lengths (torch.IntTensor): Tensor with shape (N,) representing the
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
        compact (bool, optional): Use compact layout, if True, shapes of inputs should be:
            log_probs: (STU, V)
            labels:    (SU, )
            where STU = sum(frames_lengths * (labels_lengths+1))
                  SU  = sum(labels_lengths)
    """

    assert average_frames is None or isinstance(average_frames, bool)
    assert reduction is None or reduction in ("none", "mean", "sum")
    assert isinstance(blank, int)
    assert isinstance(gather, bool)

    assert not labels.requires_grad, "labels does not require gradients"
    assert not frames_lengths.requires_grad, "frames_lengths does not require gradients"
    assert not labels_lengths.requires_grad, "labels_lengths does not require gradients"

    if compact:
        costs = RNNTLossCompact.apply(
            log_probs.float(), 
            labels, frames_lengths,
            labels_lengths, blank, 
            fastemit_lambda, 
            (log_probs.requires_grad and torch.is_grad_enabled())
        )
    else:
        if gather:

            N, T, U, V = log_probs.size()

            index = torch.full([N, T, U, 2], blank, device=labels.device, dtype=torch.long)

            index[:, :, :U-1, 1] = labels.unsqueeze(dim=1)

            log_probs = log_probs.gather(dim=3, index=index)

            blank = -1

        costs = RNNTLoss.apply(log_probs, labels, frames_lengths, labels_lengths, blank, fastemit_lambda)

    if average_frames:
        costs = costs / frames_lengths.to(log_probs)

    if reduction == "none" or reduction is None:
        return costs
    elif reduction == "sum":
        return costs.sum()
    elif reduction == "mean":
        return costs.mean()
    else:
        raise ValueError(
            f"Unknown reduction method: {reduction}, expected to be one of ['mean', 'sum', 'none']")


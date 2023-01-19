# TensorFlow binding for warp-rnnt

This package provides TensorFlow kernels that wrap the warp-rnnt library.

## Installation

Compile CUDA files:

```bash
git clone https://github.com/1ytic/warp-rnnt
cd warp-rnnt/tensorflow_binding
mkdir build && cd build && cmake  .. && make
```

Install the package into current Python environment:

```bash
python setup.py install
```

Run the tests:

```bash
python -m warp_rnnt_tf.test
```

## Using the kernels

The warp-rnnt op is available via the `warp_rnnt_tf.rnnt_loss` function:

```python
from warp_rnnt_tf import rnnt_loss
costs = rnnt_loss(log_probs, labels, frames_lengths, label_lengths)
```

See the main warp-rnnt documentation for more information.

## Python interface
```python
def rnnt_loss(
        log_probs, labels, frames_lengths, labels_lengths,
        average_frames: bool = False,
        reduction: Optional[AnyStr] = None,
        blank: int = 0,
        gather: bool = False):
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
        gather (bool, optional): Reduce memory consumption.
            Default: False.
    """
```

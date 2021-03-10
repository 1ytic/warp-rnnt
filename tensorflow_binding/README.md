# TensorFlow binding for WarpTransducer

This package provides TensorFlow kernels that wrap the WarpTransducer
library.

## Installation

To build the kernels it is necessary to have the TensorFlow source
code available, since TensorFlow doesn't currently install the
necessary headers to handle the SparseTensor that the CTCLoss op uses
to input the labels.  You can retrieve the TensorFlow source from
github.com:

```bash
git clone https://github.com/tensorflow/tensorflow.git
```
<!--
Tell the build scripts where you have the TensorFlow source tree by
setting the `TENSORFLOW_SRC_PATH` environment variable:

```bash
export TENSORFLOW_SRC_PATH=/path/to/tensorflow
```
-->
This defaults to `./build`, so from within a
new warp-rnnt clone you could build WarpTransducer like this:

```bash
mkdir build; cd build
cmake  ..
make
```

Ensure you have a GPU, you should also make sure that
`CUDA_HOME` is set to the home cuda directory (i.e. where
`include/cuda.h` and `lib/libcudart.so` live).

You should now be able to use `setup.py` to install the package into
your current Python environment:

```bash
CUDA=/path/to/cuda python setup.py install
```

You can run a few unit tests with `setup.py` as well if you want:

```bash
python tests/test_transducer_op_kernel.py
```

## Using the kernels

First import the module:

```python
import transducer_tensorflow
```

The WarpTransducer op is available via the `transducer_tensorflow.transducer_loss` function:

```python
costs = transducer_tensorflow.transducer_loss(log_probs, labels, frames_lengths, label_lengths)
```

The `log_probs` is a 4 dimensional Tensor, `labels`
is 2 dimensinal Tensor, and all the others are single dimension Tensors.
See the main WarpTransducer documentation for more information.

## Python interface
```python
def transducer_loss(
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


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
def transducer_loss(log_probs, labels, frames_lengths, labels_lengths, blank_label=0):
    '''Computes the Transducer loss between a sequence of activations and a
    ground truth labeling.
    Args:
        log_probs: A 4-D Tensor of floats.  The dimensions
                     should be (B, T, U+1, V), where B is the minibatch index,
                     T is the time index, U is the label sequence
                     length (+1 means blank label prepanded), 
                     and V indexes over activations for each 
                     symbol in the alphabet.
        labels: A 2-D Tensor of ints, a padded label sequences to make sure 
                     labels for the minibatch are same length.
        frames_lengths: A 1-D Tensor of ints, the number of time steps
                       for each sequence in the minibatch.
        labels_lengths: A 1-D Tensor of ints, the length of each label
                       for each example in the minibatch.
        blank_label: int, the label value/index that the RNNT
                     calculation should use as the blank label
    Returns:
        1-D float Tensor, the cost of each example in the minibatch
        (as negative log probabilities).
    * This class performs the softmax operation internally.
    * The label reserved for the blank symbol should be label 0.
    '''
```


# Tensorify

Prototyping with TensorFlow can often be very tideous because it provides a
limited number of possible operations on data. Pre-processing steps can be
implemented in pure Python, but if an intermediate parsing is required (or if
you want to take more advantage of TensorFlow's multi-threading capabilities)
the solution is to either write it yourself in C++ or use the `tf.py_func`
wrapper. Using the wrapper, however, requires passing a lot of arguments
to it that make software maintenance harder -- especially if you are re-using
parsing functions or classes for other pursposes.

This Python package provides `tensorflow_op`, a Python decorator that converts
regular functions into TensorFlow ops, and `tensorify`, which is a
convenience wrapper that applies the decorator to all the functions in the
given module, either in-place or in a copy. 

In short, it lets you do this:

```python
@tensorflow_op(tf.int32)
def add(a_numpy_array, another_numpy_array, extra_one=False):
    extra = 1 if extra_one else 0
    return a_numpy_array + another_numpy_array + extra

result = add(a_tf_tensor, another_tf_tensor, extra_one=True)
```

With the advantage of using any class or library you want inside the 
function. This makes wrapping existing numerical libraries for use with
Tensorflow extremely easy.

**WARNING**: All functions wrapped with tensorify will be executed in the
same machine as the main script, so this is not a very efficient solution.
This is intended for prototyping and for pre-processing or parsing code only.
Do not implement heavy deep learning layers with this!

## Installation

The tensorify package is not yet on the PyPI archive; you can install it with:

```bash
pip install git+https://github.com/lemonzi/tensorify
```

And then import it as usual.

## Usage

### `tensorflow_op`

> A decorator that takes a function and turns it into a TensorFlow op.

A call to the decorated function will create a node with input tensors
set to the function arguments, and the supplied key-word arguments will
be passed to the function directly using a partial invocation.

The decorator takes as optional arguments a list with output types, a
name for the op, and whether the function is stateful (`False` by default).

If no name is supplied, the CamelCased name of the function will be used.
The name can also be modified at call time using the `with_name()` method:

```python
@tensorflow_op(tf.int32)
def add(x, y, extra_one=False):
    return x + y + (1 if extra_one else 0)

input_one = tf.constant([1])
input_two = tf.constant([2])
answer.with_name("AddPlusOne")(input_one, input_two, extra_one=True)
answer.with_name("AddAndThatsIt")(input_one, input_two)
```

Similarly, the op can be made stateful using `stateful()`. In this example,
we need to supply `is_method` so that the `self` argument doesn't get
sent to the TensorFlow engine and stays in the closure instead.
Actually, when `stateful` is not provided as argument and `is_method` is
true the op is stateful by default; this is only an example.

```python
class Accumulator:
    def __init__(self):
        self.state = 0
    @tensorify.tensorflow_op(tf.int64, is_method=True)
    def accumulate(self, x):
        self.state = self.state + x
        return self.state

x = tf.constant([1])
y = tf.constant([2])
my_accumulator = Accumulator()
after_accumulating_x = my_accumulator.stateful()(x)
after_accumulating_y = my_accumulator.stateful()(y)
```

Notice that, in this example, all ops that come form the same accumulator
will accumulate to the same buffer, so the accumulator acts as a global
counter. This is not very efficient, though; it would be better to use a
`tf.Variable` for state storage and have the operation be stateless.

The same applies for the outputs:

```python
@tensorflow_op()
def replicate(value, how_many_times):
    return [value] * how_many_times

x = tf.constant([1])
three_x = replicate.with_outputs([tf.int32] * 3)(x, 3)
```

### `tensorify`

> Converts all functions in a module into TensorFlow ops.

Example usage:

In `fancy_module.py`:

```python
def add(x, y):
    return x + y
```

In `app.py`:

```python
import tensorflow as tf
from tensorify import tensorify

import fancy_module
tensorify(fancy_module, tf.int32)

x_tensor = tf.constant([1])
y_tensor = tf.constant([2])
result = fancy_module.add(x_tensor, y_tensor)
```

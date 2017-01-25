# Tensorify

This Python package provides `tensorflow_op`, a Python decorator that converts
regular functions into TensorFlow ops, and `tensorify`, which is a
convenience wrapper that applies the decorator to all the functions in the
given module, either in-place or in a copy. 

## `tensorflow_op`

> A decorator that takes a function and turns it into a TensorFlow op.

A call to the decorated function will create a node with input tensors
set to the function arguments, and the supplied key-word arguments will
be passed to the function directly using a partial invocation.

The decorator takes as optional arguments a list with output types, a
name for the op, and whether the function is stateful (`False` by default).

If no name is supplied, the CamelCased name of the function will be used.
The name can also be modified at call time using the `with_name()` method:

```python
@tensorflow_op([tf.int32], name="Add")
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

## `tensorify`

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

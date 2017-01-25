"""Utils that help in wrapping Python functions as TensorFlow ops.

Quim Llimona, 2017.
"""

import copy
import functools
import inspect

import tensorflow as tf


def tensorflow_op(outputs=[], stateful=None, name=None, is_method=False):
    """A decorator that takes a function and turns it into a TensorFlow op.

    A call to the decorated function will create a node with input tensors
    set to the function arguments, and the supplied key-word arguments will
    be passed to the function directly using a partial invocation.

    The decorator takes as optional arguments a list with output types, a
    name for the op, and whether the function is stateful (`False` by default).

    If no name is supplied, the CamelCased name of the function will be used.
    The name can also be modified at call time using the `with_name()` method:

        @tensorflow_op([tf.int32], name="Add")
        def add(x, y, extra_one=False):
            return x + y + (1 if extra_one else 0)

        input_one = tf.constant([1])
        input_two = tf.constant([2])
        answer.with_name("AddPlusOne")(input_one, input_two, extra_one=True)
        answer.with_name("AddAndThatsIt")(input_one, input_two)

    Similarly, the op can be made stateful using `stateful()`. In this example,
    we need to supply `is_method` so that the `self` argument doesn't get
    sent to the TensorFlow engine and stays in the closure instead.
    Actually, when `stateful` is not provided as argument and `is_method` is
    true the op is stateful by default; this is only an example.

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

    Notice that, in this example, all ops that come form the same accumulator
    will accumulate to the same buffer, so the accumulator acts as a global
    counter. This is not very efficient, though; it would be better to use a
    `tf.Variable` for state storage and have the operation be stateless.

    The same applies for the outputs:

        @tensorflow_op()
        def replicate(value, how_many_times):
            return [value] * how_many_times

        x = tf.constant([1])
        three_x = replicate.with_outputs([tf.int32] * 3)(x, 3)
    """
    # Class methods are usually stateful.
    if stateful is None:
        stateful = is_method
    # This is the function that returns a decorated function when called.
    # It is normally called internally by Python when using @tensorflow_op.
    def tf_decorator(function):
        # TODO: Fix is_method and remove it from the argument list.
        # is_method = inspect.ismethod(function)
        # This function what will actually be called when the user calls the
        # returned function.
        @functools.wraps(function)
        def tf_wrapper(*args, **kwargs):
            # TODO(lemonzi): Detect if one of the arguments is self.
            # Local, mutable copy of the op name
            name_to_use = name
            # If we have no name, we figure it out from the function name.
            if name_to_use is None:
                name_to_use = camel_case(function.__name__)
            # If the function is a class method, strip out the self object.
            if is_method:
                self = args[0]
                args = args[1:]
                partial_function = functools.partial(function, self, **kwargs)
            else:
                partial_function = functools.partial(function, **kwargs)
            # Wrap a partial application of the function as a TF op.
            return tf.py_func(partial_function, args, outputs,
                              stateful=stateful, name=name_to_use)
        # Returns a new decorated function with a different op name.
        def set_name(new_name):
            return tensorflow_op(outputs=outputs, stateful=stateful,
                                 name=new_name, is_method=is_method)(function)
        # Returns a new decorated function with a different statefulness.
        def set_stateful(new_statefulness=True):
            return tensorflow_op(outputs=outputs, stateful=new_stateful,
                                 name=name, is_method=is_method)(function)
        # Returns a new decorated function with a different output signature.
        def set_outputs(new_outputs):
            return tensorflow_op(outputs=new_outputs, stateful=stateful,
                                 name=name, is_method=is_method)(function)
        # The three functions are exposed publicly as attributes.
        tf_wrapper.with_name = set_name
        tf_wrapper.stateful = set_stateful
        tf_wrapper.with_outputs = set_outputs
        return tf_wrapper
    return tf_decorator


def camel_case(name):
    """Converts the given name in snake_case or lowerCamelCase to CamelCase."""
    words = name.split('_')
    return ''.join(word.capitalize() for word in words)


def tensorify(module, default_outputs, in_place=True):
    """Converts all functions inside a module into TensorFlow ops."""
    new_module = module if in_place else copy.copy(module)
    functions = inspect.getmembers(new_module, inspect.isfunction)
    for name, function in functions:
        operized_function = tensorflow_op(outputs=default_outputs)(function)
        new_module.__dict__[name] = operized_function
    return new_module

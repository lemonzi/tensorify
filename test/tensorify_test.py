"""A small test for tensorify.

Quim Llimona, 2017.
"""

import tensorflow as tf
import tensorify

import functions
tensorify.tensorify(functions, tf.int64)


class Accumulator:
    def __init__(self):
        self.state = 0
    @tensorify.tensorflow_op(tf.int64, is_method=True)
    def accumulate(self, x):
        self.state = self.state + x
        return self.state


class TensorflowOpTest(tf.test.TestCase):
    def testBasic(self):
        with self.test_session():
            self.assertAllEqual(functions.add(2, 3).eval(), 5)
    def testKeyword(self):
        with self.test_session():
            self.assertAllEqual(functions.add(2, 3, extra_one=True).eval(), 6)
    def testStateful(self):
        with self.test_session():
            my_accumulator = Accumulator()
            one = my_accumulator.accumulate(1)
            self.assertAllEqual(one.eval(), 1)
            self.assertAllEqual(one.eval(), 2)
            two = my_accumulator.accumulate(2)
            self.assertAllEqual(two.eval(), 4)
    def testMultipleOutput(self):
        with self.test_session():
            replicas = functions.replicate.with_outputs([tf.int32] * 3)(2, 3)
            self.assertEqual(len(replicas), 3)
            for replica in replicas:
                self.assertAllEqual(replica.eval(), 2)


if __name__ == '__main__':
    tf.test.main()

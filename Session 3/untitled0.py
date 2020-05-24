# -*- coding: utf-8 -*-
"""
Created on Fri May 22 09:26:08 2020

@author: Linh LV
"""
#in TF2, Session() has been removed, so that's no longer necessary in TF2

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

x1 = tf.constant(5)
x2 = tf.constant(6)

result = tf.multiply(x1, x2)

with tf.compat.v1.Session() as sess:
    output = sess.run(result)
    print (output)
#sess = tf.compat.v1.Session()
#output = sess.run(result)
#print (output)
#sess.close()
from Config import *

class SmartQueue(object):
    def __init__(self, config = None, capacity=100, name_list=None):
        self.config = config
        self.enqueue_placeholder = tf.placeholder(dtype=tf.string)
        self.filename_queue = tf.FIFOQueue(capacity, tf.string)
        if name_list != None:
            self.filename_queue.enqueue_many(name_list)
        self.enqueue_op = self.filename_queue.enqueue(self.enqueue_placeholder)
        self.dequeue_op = self.filename_queue.dequeue()

    def enqueue(self, sess, x):
        sess.run(self.enqueue_op, feed_dict={self.enqueue_placeholder:x})

    def dequeue(self, sess):
        x = sess.run(self.dequeue_op)
        self.enqueue(sess, x)

import numpy
import tensorflow as tf

class DQNetwork:
    def __init__(self, state_size, action_size, learning_rate, name="DQNetwork"):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, self.action_size], name="actions")
            self.targetQ = tf.placehodler(tf.float32, [None], name="targetQ")

            """ 1st ConvNet: CNN, ELU """
            self.conv1 = tf.layers.conv2d(inputs = self.inputs_,
                    filters = 32,
                    kernel_size = [8,8],
                    strides = [4,4],
                    padding = "VALID",
                    kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                    name="conv1")
            self.conv1_out = tf.nn.elu(self.conv1, name="conv1_out")

            """ 2nd ConvNet: CNN, ELU """
            self.conv2 = tf.layers.conv2d(inputs = self.conv1_out,
                    filters = 64,
                    kernel_size = [4,4],
                    strides = [2,2],
                    padding = "VALID",
                    kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                    name="conv2d")
            self.conv2_out = tf.nn.elu(self.conv2, name="conv2_out")



#!/usr/bin/env python3
#-*- coding: utf-8 -*-
'''
Source:
-------
<https://gist.github.com/nivwusquorum/b18ce332bde37e156034e5d3f60f8a23>

Short and sweet LSTM implementation in Tensorflow.
Motivation:
When Tensorflow was released, adding RNNs was a bit of a hack - it required
building separate graphs for every number of timesteps and was a bit obscure
to use. Since then TF devs added things like `dynamic_rnn`, `scan` and `map_fn`.
Currently the APIs are decent, but all the tutorials that I am aware of are not
making the best use of the new APIs.
Advantages of this implementation:
- No need to specify number of timesteps ahead of time. Number of timesteps is
  inferred from shape of input tensor. Can use the same graph for multiple
  different numbers of timesteps.
- No need to specify batch size ahead of time. Batch size is inferred from shape
  of input tensor. Can use the same graph for multiple different batch sizes.
- Easy to swap out different recurrent gadgets (RNN, LSTM, GRU, your new
  creative idea)
'''


import random

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn, layers


NUM_EPOCHS = 5
NUM_BITS = 10
NUM_ITER = 100
TINY = np.finfo(np.float32).eps  # 1e-6                         # pylint: disable=no-member
BATCH_SIZE = 16

INPUT_SIZE = 2          # 2 bits per timestep
RNN_HIDDEN = 20
OUTPUT_SIZE = 1         # 1 bit per timestep
LEARNING_RATE = 0.01

USE_LSTM = True

SEED = 2


random.seed(SEED)
# np.random.seed(SEED)
tf.set_random_seed(SEED)

##########################################################################
##                           DATASET GENERATION                               ##
##                                                                            ##
##  The problem we are trying to solve is adding two binary numbers. The      ##
##  numbers are reversed, so that the state of RNN can add the numbers        ##
##  perfectly provided it can learn to store carry in the state. Timestep t   ##
##  corresponds to bit len(number) - t.                                       ##
##########################################################################

def as_bytes(num, final_size):
    res = []
    for _ in range(final_size):
        res.append(num % 2)
        num //= 2
    return res


def generate_example(num_bits):
    num_a = random.randint(0, 2**(num_bits - 1) - 1)
    num_b = random.randint(0, 2**(num_bits - 1) - 1)
    res = num_a + num_b
    return (as_bytes(num_a, num_bits),
            as_bytes(num_b, num_bits),
            as_bytes(res, num_bits))


def generate_batch(num_bits, batch_size):
    '''Generates instance of a problem.
    Returns
    -------
    x: np.array
        two numbers to be added represented by bits.
        shape: b, i, n
        where:
            b is bit index from the end
            i is example idx in batch
            n is one of [0,1] depending for first and
                second summand respectively
    y: np.array
        the result of the addition
        shape: b, i, n
        where:
            b is bit index from the end
            i is example idx in batch
            n is always 0
    '''
    x = np.empty((num_bits, batch_size, 2))
    y = np.empty((num_bits, batch_size, 1))

    for inst in range(batch_size):
        num_a, num_b, res = generate_example(num_bits)
        x[:, inst, 0] = num_a
        x[:, inst, 1] = num_b
        y[:, inst, 0] = res

    return x, y


##########################################################################
##                           GRAPH DEFINITION                           ##
##########################################################################

def build_graph():
    inputs = tf.placeholder(tf.float32, shape=(None, None, INPUT_SIZE))  # (time, batch, in)
    labels = tf.placeholder(tf.float32, shape=(None, None, OUTPUT_SIZE))  # (time, batch, out)

    # Here cell can be any function you want, provided it has two attributes:
    #     - cell.zero_state(batch_size, dtype)- tensor which is an initial value
    #                                           for state in __call__
    #     - cell.__call__(input, state) - function that given input and previous
    #                                     state returns tuple (output, state) where
    #                                     state is the state passed to the next
    #                                     timestep and output is the tensor used
    #                                     for infering the output at timestep. For
    #                                     example for LSTM, output is just hidden,
    #                                     but state is memory + hidden
    # Example LSTM cell with learnable zero_state can be found here:
    #    https://gist.github.com/nivwusquorum/160d5cf7e1e82c21fad3ebf04f039317
    if USE_LSTM:
        cell = rnn.BasicLSTMCell(RNN_HIDDEN, state_is_tuple=True)
    else:
        cell = rnn.BasicRNNCell(RNN_HIDDEN)

    # Create initial state. Here it is just a constant tensor filled with zeros,
    # but in principle it could be a learnable parameter. This is a bit tricky
    # to do for LSTM's tuple state, but can be achieved by creating two vector
    # Variables, which are then tiled along batch dimension and grouped into tuple.
    batch_size = tf.shape(inputs)[1]
    initial_state = cell.zero_state(batch_size=batch_size, dtype=tf.float32)

    # Given inputs (time, batch, input_size) outputs a tuple
    #  - outputs: (time, batch, output_size)  [do not mistake with OUTPUT_SIZE]
    #  - states:  (time, batch, hidden_size)
    rnn_outputs, _rnn_states = tf.nn.dynamic_rnn(cell, inputs,
                                                 initial_state=initial_state, time_major=True)

    # project output from rnn output size to OUTPUT_SIZE. Sometimes it is worth adding
    # an extra layer here.


    def final_projection(x):
        return layers.fully_connected(x, num_outputs=OUTPUT_SIZE, activation_fn=tf.nn.sigmoid)


    # apply projection to every timestep.
    prediction = tf.map_fn(fn=final_projection, elems=rnn_outputs)

    labels_sqz = tf.transpose(tf.squeeze(labels[:, :, 0]))
    predic_sqz = tf.transpose(tf.squeeze(prediction[:, :, 0]))

    # compute elementwise cross entropy.
    loss_tsr = -(labels_sqz * tf.log(predic_sqz + TINY) +
                 (1.0 - labels_sqz) * tf.log(1.0 - predic_sqz + TINY))

    # loss_tsr = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_sqz, logits=predic_sqz,
    #                                                    name='loss')
    # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=prediction,
    #                                                       name='loss')
    # loss_tsr = tf.nn.softmax_cross_entropy_with_logits(labels=labels_sqz, logits=predic_sqz,
    #                                                    name='loss')

    loss = tf.reduce_mean(loss_tsr)

    # optimize
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

    # assuming that absolute difference between output and correct answer is 0.5
    # or less we can round it to the correct output.
    accuracy = tf.reduce_mean(tf.cast(tf.abs(labels - prediction) < 0.5, tf.float32))

    return (loss, optimizer, inputs, labels, accuracy, prediction, labels_sqz, predic_sqz, loss_tsr)


##########################################################################
##                           TRAINING LOOP                              ##
##########################################################################

def run_training():
    valid_x, valid_y = generate_batch(num_bits=NUM_BITS, batch_size=BATCH_SIZE)

    (loss, optimizer, inputs, labels, accuracy, prediction, labels_sqz, predic_sqz, loss_tsr
    ) = build_graph()

    session = tf.Session()
    # For some reason it is our job to do this:
    session.run(tf.global_variables_initializer())

    for epoch in range(NUM_EPOCHS):
        epoch_error = 0
        for _ in range(NUM_ITER):
            # here optimizer is what triggers backprop. loss and accuracy on their
            # own do not trigger the backprop.
            x, y = generate_batch(num_bits=NUM_BITS, batch_size=BATCH_SIZE)
            epoch_error += session.run(fetches=[loss, optimizer],
                                       feed_dict={inputs: x, labels: y})[0]

        epoch_error /= NUM_ITER
        valid_accuracy, predic, labels_arr, predic_arr, loss_arr = session.run(
            fetches=[accuracy, prediction, labels_sqz, predic_sqz, loss_tsr],
            feed_dict={inputs: valid_x, labels: valid_y})

        print("Epoch {} - loss: {:.2f} - valid accuracy: {:5.1f}% - predic: {}"#{:.6f}"
              "".format(epoch, epoch_error, valid_accuracy * 100.0, loss_arr.shape))

        np.set_printoptions(precision=2)
        np.set_printoptions(suppress=True)  # Suppress the use of scientific notation
        print(labels_arr)
        print(predic_arr)
        print(loss_arr)

def main():
    run_training()

if __name__ == '__main__':
    main()

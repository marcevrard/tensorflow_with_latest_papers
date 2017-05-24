#!/usr/bin/env python3
#-*- coding: utf-8 -*-

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

'''
Example / benchmark for building a PTB LSTM model.
==================================================
Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329

There are 3 supported model configurations:
-------------------------------------------
| config | epochs | train | valid  | test   |
|:-------|:------:|------:|-------:|-------:|
| small  | 13     | 37.99 | 121.39 | 115.91 |
| medium | 39     | 48.45 |  86.16 |  82.07 |
| large  | 55     | 37.87 |  82.62 |  78.29 |

The exact results may vary depending on the random initialization.

The hyperparameters used in the model:
--------------------------------------
- `init_scale` - the initial scale of the weights
- `lr` - the initial value of the learning rate
- `max_grad` - the maximum permissible norm of the gradient
- `num_layers` - the number of LSTM layers
- `num_steps` - the number of unrolled steps of LSTM
- `hidden_size` - the number of LSTM units
- `lr_max_epoch` - the number of epochs trained with the initial learning rate
- `max_epoch` - the total number of epochs for training
- `keep_prob` - the probability of keeping weights in the dropout layer
- `lr_decay` - the decay of the learning rate for each epoch after 'lr_max_epoch'
- `batch_size` - the batch size

The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:
-----------------------------------------
    $ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
    $ tar xvf simple-examples.tgz

To run:
-------
    $ ./ptb_word_lm.py --data_path=simple-examples/data/
'''
import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.contrib import framework, legacy_seq2seq, rnn

import rnn_cell_layernorm_modern            # pylint: disable=unused-import
import rnn_cell_modern                      # pylint: disable=unused-import
import rnn_cell_mulint_layernorm_modern     # pylint: disable=unused-import
import rnn_cell_mulint_modern               # pylint: disable=unused-import

from misc_tools import load_config
from print_tools import logging_handler
from tf_ptb_model_util import reader


DATA_TYPE = tf.float32


class PTBInput:
    '''The input data.'''

    def __init__(self, config, data, name=None):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.input_data, self.targets = reader.ptb_producer(
            data, batch_size, num_steps, name=name)


class PTBModel:
    '''The PTB model.'''

    def __init__(self, is_training, config, input_):
        self._input = input_

        batch_size = input_.batch_size
        num_steps = input_.num_steps
        size = config.hidden_size
        vocab_size = config.vocab_size

        def cell():
            # return rnn.BasicLSTMCell(num_units=size, forget_bias=1.0,
            #                          reuse=tf.get_variable_scope().reuse)
            return rnn_cell_modern.HighwayRNNCell(num_units=size)
        # cell = rnn_cell_modern.JZS1Cell(size)
        # cell = rnn_cell_mulint_modern.BasicRNNCell_MulInt(size)
        # cell = rnn_cell_mulint_modern.GRUCell_MulInt(size)
        # cell = rnn_cell_mulint_modern.BasicLSTMCell_MulInt(size)
        # cell = rnn_cell_mulint_modern.HighwayRNNCell_MulInt(size)
        # cell = rnn_cell_mulint_layernorm_modern.BasicLSTMCell_MulInt_LayerNorm(size)
        # cell = rnn_cell_mulint_layernorm_modern.GRUCell_MulInt_LayerNorm(size)
        # cell = rnn_cell_mulint_layernorm_modern.HighwayRNNCell_MulInt_LayerNorm(size)
        # cell = rnn_cell_layernorm_modern.BasicLSTMCell_LayerNorm(size)
        # cell = rnn_cell_layernorm_modern.GRUCell_LayerNorm(size)
        # cell = rnn_cell_layernorm_modern.HighwayRNNCell_LayerNorm(size)
        # cell = rnn_cell_modern.LSTMCellMemoryArray(
        #     size, num_memory_arrays=2,
        #     use_multiplicative_integration=True, use_recurrent_dropout=False)
        # cell = rnn_cell_modern.MGUCell(
        #     size, use_multiplicative_integration=True, use_recurrent_dropout=False)

        op_cell = cell
        if is_training and config.keep_prob < 1:
            def op_cell():
                return rnn.DropoutWrapper(cell(), output_keep_prob=config.keep_prob)
        multi_cell = rnn.MultiRNNCell([op_cell() for _ in range(config.num_layers)])

        # multi_cell = rnn.MultiRNNCell([cell for _ in range(config.num_layers)])
        # # multi_cell = rnn.MultiRNNCell([cell] * config.num_layers)

        self._initial_state = multi_cell.zero_state(batch_size, dtype=DATA_TYPE)

        # with tf.device('/cpu:0'):
        embedding = tf.get_variable('embedding', shape=[vocab_size, size])
        inputs = tf.nn.embedding_lookup(embedding, ids=input_.input_data)

        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, keep_prob=config.keep_prob)

        # Simplified version of tensorflow.models.rnn.rnn.py's rnn().
        # This builds an unrolled LSTM for tutorial purposes only.
        # In general, use the rnn() or state_saving_rnn() from rnn.py.
        #
        # The alternative version of the code below is:
        #
        # from tensorflow.models.rnn import rnn
        # inputs = [tf.squeeze(input_, [1])
        #           for input_ in tf.split(1, num_steps, inputs)]
        # outputs, state = rnn.rnn(multi_cell, inputs, initial_state=self._initial_state)
        outputs = []
        state = self._initial_state
        with tf.variable_scope('RNN'):
            for time_step in range(num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = multi_cell(inputs=inputs[:, time_step, :], state=state)
                outputs.append(cell_output)

        output = tf.reshape(tf.stack(values=outputs, axis=1),
                            shape=[-1, size])
        # softmax_w = tf.transpose(embedding)  # weight tying
        softmax_w = tf.get_variable('softmax_w', shape=[size, vocab_size], dtype=DATA_TYPE)
        softmax_b = tf.get_variable('softmax_b', shape=[vocab_size], dtype=DATA_TYPE)
        logits = tf.matmul(output, softmax_w) + softmax_b
        loss = legacy_seq2seq.sequence_loss_by_example(
            logits=[logits],
            targets=[tf.reshape(input_.targets, shape=[-1])],
            weights=[tf.ones(shape=[batch_size * num_steps], dtype=DATA_TYPE)])

        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._final_state = state

        if not is_training:
            return

        self._lr = tf.Variable(initial_value=0.0, trainable=False)  # TODO: try None as init val?
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          clip_norm=config.max_grad)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)     # TODO: try momentum
        # optimizer = tf.train.AdamOptimizer(self._lr)

        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=framework.get_or_create_global_step())

        self._new_lr = tf.placeholder(DATA_TYPE, shape=[], name='new_learning_rate')
        self._lr_update = tf.assign(ref=self._lr, value=self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(fetches=self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def input(self):
        return self._input

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op


def run_epoch(session, model, eval_op=None, verbose=False):
    '''Runs the model on the given data.'''
    start_time = time.time()
    costs = 0.0
    iters = 0

    state = session.run(fetches=model.initial_state)

    fetches = {'cost': model.cost,
               'final_state': model.final_state}

    if eval_op is not None:
        fetches['eval_op'] = eval_op

    for step in range(model.input.epoch_size):
        feed_dict = {}
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        vals = session.run(fetches, feed_dict)
        cost = vals['cost']
        state = vals['final_state']

        costs += cost
        iters += model.input.num_steps

        if verbose and step % (model.input.epoch_size // 10) == 10:
            logging.info(
                "{:.1f} perplexity: {:7.2f} speed: {:.0f} wps"
                "".format(step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
                          iters * model.input.batch_size / (time.time() - start_time)))

    return np.exp(costs / iters)


def main(argp):

    config_fpath = os.path.join('./configs', argp.model + '_config.json')
    config = load_config(config_fpath, extra_config=argp.update_config, ntpl=True)
    eval_config = load_config(config_fpath, extra_config=dict(argp.update_config,
                                                              **{"batch_size": 1,
                                                                 "num_steps": 1}), ntpl=True)

    logging_handler(log_fpath=os.path.join('./logs', config.config_txt))

    logging.info(config)
    logging.info("Configuration: {}".format(argp.model))

    raw_data = reader.ptb_raw_data(argp.data_path)
    train_data, valid_data, test_data, _ = raw_data

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(minval=-config.init_scale,
                                                    maxval=config.init_scale)

        with tf.name_scope("Train"):
            train_input = PTBInput(config=config, data=train_data, name="TrainInput")
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                model = PTBModel(is_training=True, config=config, input_=train_input)
            tf.summary.scalar("Training_Loss", model.cost)
            tf.summary.scalar("Learning_Rate", model.lr)

        with tf.name_scope("Valid"):
            valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                model_valid = PTBModel(is_training=False, config=config, input_=valid_input)
            tf.summary.scalar("Validation_Loss", model_valid.cost)

        with tf.name_scope("Test"):
            test_input = PTBInput(config=eval_config, data=test_data, name="TestInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                model_test = PTBModel(is_training=False, config=eval_config, input_=test_input)

        sv = tf.train.Supervisor(logdir=argp.save_path)
        with sv.managed_session() as session:

            # epoch = 0   # To avoid warning after loop block
            for epoch in range(1, config.max_epoch + 1):
                lr_decay = config.lr_decay ** max(epoch - config.lr_max_epoch, 0.0)
                model.assign_lr(session, lr_value=config.lr * lr_decay)

                logging.info(
                    "Epoch: {} Learning rate: {:.4f}".format(epoch, session.run(model.lr)))
                train_perplexity = run_epoch(session, model, eval_op=model.train_op,
                                             verbose=True)

                logging.info(
                    "Epoch: {} Train Perplexity: {:.2f}".format(epoch, train_perplexity))
                valid_perplexity = run_epoch(session, model_valid)
                logging.info(
                    "Epoch: {} Valid Perplexity: {:.2f}".format(epoch, valid_perplexity))

            test_perplexity = run_epoch(session, model_test)
            logging.info("Test Perplexity: {:.2f}".format(test_perplexity))

            # if argp.save_path:
            #     model_fpath = os.path.join(argp.save_path, "model-epoch{}".format(epoch))
            #     logging.info("Saving model to {}".format(model_fpath))
            #     sv.saver.save(session, model_fpath, global_step=sv.global_step)


def get_args(args=None):     # Add possibility to manually insert args at runtime (for ipynb)

    parser = argparse.ArgumentParser(description="Select options to run the process.")
    parser.add_argument('--data-path', default='./simple-examples/data',
                        help='Choose the data path.')
    parser.add_argument('--model', choices=['small', 'medium', 'large', 'test', 'highway'],
                        default='small', help='Choose the size of model to train.')
    parser.add_argument('--save-path', default='./models',
                        help='Model output directory.')
    parser.add_argument('-u', '--update-config', type=json.loads, default='{}',
                        help="Update configuration setting(s).")

    return parser.parse_args(args)


if __name__ == '__main__':
    try:
        main(get_args())
    except KeyboardInterrupt:
        sys.exit("\nProgram interrupted by user.\n")

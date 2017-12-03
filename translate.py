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

"""Binary for training translation models and decoding from them.

Running this program without --decode will download the WMT corpus into
the directory specified as --data_dir and tokenize it in a very basic way,
and then start training a model saving checkpoints to --train_dir.

Running with --decode starts an interactive loop so you can see how
the current checkpoint translates English sentences into French.

See the following papers for more information on neural translation models.
 * http://arxiv.org/abs/1409.3215
 * http://arxiv.org/abs/1409.0473
 * http://arxiv.org/abs/1412.2007
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import data_utils
import seq2seq_model


tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
# make batch size smaller since I don't have that much data
tf.app.flags.DEFINE_integer("batch_size", 16,
                            "Batch size to use during training.")

# This many neurons per layer should be enough
tf.app.flags.DEFINE_integer("size", 128, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 2, "Number of layers in the model.")

# Assumption: Instruction vocab size is bigger, but command vocab is pretty small
tf.app.flags.DEFINE_integer("from_vocab_size", 40000, "English vocabulary size, big.")
tf.app.flags.DEFINE_integer("to_vocab_size", 4000, "Command vocab size, smaller.")

# places where I save my data and store trained parameters
tf.app.flags.DEFINE_string("data_dir", "data", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "parameters", "Training directory.")

# the data and labels, data refers to english instructions, labels refer to commands
tf.app.flags.DEFINE_string("from_train_data", 'data/data.txt', "Training data.")
tf.app.flags.DEFINE_string("to_train_data", 'data/label.txt', "Training data.")

tf.app.flags.DEFINE_string("from_dev_data", 'data/validation/data.txt', "Training data.")
tf.app.flags.DEFINE_string("to_dev_data", 'data/validation/label.txt', "Training data.")

tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 20,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_boolean("use_fp16", False,
                            "Train using fp16 instead of fp32.")

FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [(50, 50)]


def read_data(source_path, target_path, max_size=None):
  """Read data from source and target files and put into buckets.

  Args:
    source_path: path to the files with token-ids for the source language.
    target_path: path to the file with token-ids for the target language;
      it must be aligned with the source file: n-th line contains the desired
      output for n-th line from the source_path.
    max_size: maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).

  Returns:
    data_set: a list of length len(_buckets); data_set[n] contains a list of
      (source, target) pairs read from the provided data files that fit
      into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
      len(target) < _buckets[n][1]; source and target are lists of token-ids.
  """
  data_set = [[] for _ in _buckets]
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      source, target = source_file.readline(), target_file.readline()
      counter = 0
      while source and target and (not max_size or counter < max_size):
        counter += 1
        if counter % 100000 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()
        source_ids = [int(x) for x in source.split()]
        target_ids = [int(x) for x in target.split()]
        target_ids.append(data_utils.EOS_ID)
        for bucket_id, (source_size, target_size) in enumerate(_buckets):
          if len(source_ids) < source_size and len(target_ids) < target_size:
            data_set[bucket_id].append([source_ids, target_ids])
            break
        source, target = source_file.readline(), target_file.readline()
  return data_set


def create_model(session, forward_only):
  """Create translation model and initialize or load parameters in session."""
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  model = seq2seq_model.Seq2SeqModel(
      FLAGS.from_vocab_size,
      FLAGS.to_vocab_size,
      _buckets,
      FLAGS.size,
      FLAGS.num_layers,
      FLAGS.max_gradient_norm,
      FLAGS.batch_size,
      FLAGS.learning_rate,
      FLAGS.learning_rate_decay_factor,
      forward_only=forward_only,
      dtype=dtype)
  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.global_variables_initializer())
  return model


def train():
  """Train a en->fr translation model using WMT data."""
  from_train = None
  to_train = None
  from_dev = None
  to_dev = None
  if FLAGS.from_train_data and FLAGS.to_train_data:
    from_train_data = FLAGS.from_train_data
    to_train_data = FLAGS.to_train_data
    from_dev_data = from_train_data
    to_dev_data = to_train_data
    if FLAGS.from_dev_data and FLAGS.to_dev_data:
      from_dev_data = FLAGS.from_dev_data
      to_dev_data = FLAGS.to_dev_data
    from_train, to_train, from_dev, to_dev, _, _ = data_utils.prepare_data(
        FLAGS.data_dir,
        from_train_data,
        to_train_data,
        from_dev_data,
        to_dev_data,
        FLAGS.from_vocab_size,
        FLAGS.to_vocab_size,
        data_utils.custom_tokenizer)
  else:
      # nothing to train, just return
      return

  with tf.Session() as sess:
    # Create model.
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
    model = create_model(sess, False)

    # Read data into buckets and compute their sizes.
    print ("Reading development and training data (limit: %d)."
           % FLAGS.max_train_data_size)
    dev_set = read_data(from_dev, to_dev)
    #print("{}".format(len(dev_set[0])))
    #print("{}".format(dev_set[0]))
    train_set = read_data(from_train, to_train, FLAGS.max_train_data_size)
    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
    train_total_size = float(sum(train_bucket_sizes))

    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in xrange(len(train_bucket_sizes))]

    # This is the training loop.
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []
    eval_prev_ppx = []
    while True:
      # Choose a bucket according to data distribution. We pick a random number
      # in [0, 1] and use the corresponding interval in train_buckets_scale.
      random_number_01 = np.random.random_sample()
      bucket_id = min([i for i in xrange(len(train_buckets_scale))
                       if train_buckets_scale[i] > random_number_01])

      # Get a batch and make a step.
      start_time = time.time()
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          train_set, bucket_id)

      _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                   target_weights, bucket_id, False)
      step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
      loss += step_loss / FLAGS.steps_per_checkpoint
      current_step += 1
      print("step: {}".format(current_step))
      # Once in a while, we save checkpoint, print statistics, and run evals.
      if current_step % FLAGS.steps_per_checkpoint == 0:
        # Print statistics for the previous epoch.
        perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
        print ("global step %d learning rate %.4f step-time %.2f perplexity "
               "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                         step_time, perplexity))

        # Decrease learning rate if no improvement was seen over last 3 times.
        if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
          sess.run(model.learning_rate_decay_op)
        previous_losses.append(loss)
        # Save checkpoint and zero timer and loss.
        checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        step_time, loss = 0.0, 0.0
        # Run evals on development set and print their perplexity.
        bucket_id = 0
        if len(dev_set[bucket_id]) == 0:
          print("  eval: empty bucket %d" % (bucket_id))
          continue
        encoder_inputs, decoder_inputs, target_weights = model.get_batch(
            dev_set, bucket_id)

        # forward_only set to True, we are not learning the validation set
        _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                     target_weights, bucket_id, True)
        eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float(
            "inf")
        # Early Stopping
        #if perplexity < 3 and len(eval_prev_ppx) > 9 and eval_ppx > max(eval_prev_ppx[-10:]):
        #  break

        eval_prev_ppx.append(eval_ppx)

        print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
        sys.stdout.flush()

def check_order(a, b):
  for i in range(len(a) - 1):
    try:
      i1 = b.index(a[i])
      i2 = b.index(a[i+1])

      b.remove(a[i])

      if i2 < i1:
        return False
    except ValueError:
      continue
  return True


def decode_helper(data, label, size, set_name, from_vocab, to_vocab, model, sess):
  count_complete_match = 0
  count_right_order = 0
  count_portion_match = []
  for dd, ll in zip(data, label):
    dd = dd.strip()
    ll = ll.strip()
    token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(dd), from_vocab)
    # Which bucket does it belong to?
    bucket_id = len(_buckets) - 1
    for i, bucket in enumerate(_buckets):
      if bucket[0] >= len(token_ids):
        bucket_id = i
        break
    else:
      logging.warning("Sentence truncated: %s", dd)

    # Get a 1-element batch to feed the sentence to the model.
    encoder_inputs, decoder_inputs, target_weights = model.get_batch(
        {bucket_id: [(token_ids, [])]}, bucket_id)
    # Get output logits for the sentence.
    _, one_loss, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                     target_weights, bucket_id, True)
    # This is a greedy decoder - outputs are just argmaxes of output_logits.
    outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
    # print("output: {}".format(outputs[0]))
    # If there is an EOS symbol in outputs, cut them at that point.
    if data_utils.EOS_ID in outputs:
      outputs = outputs[:outputs.index(data_utils.EOS_ID)]

    # compare the target label with our output
    target_label = data_utils.sentence_to_token_ids(tf.compat.as_bytes(ll), to_vocab)
    # find the perplexity of this one element prediction
    one_ppx = math.exp(float(one_loss)) if one_loss < 300 else float(
            "inf")
    
    # num of parts of an instruction that matched
    count_num_same = 0
    for target_label_v, outputs_v in zip(target_label, outputs):
      if target_label_v == outputs_v:
        count_num_same += 1
    count_portion_match.append(count_num_same*1.0/len(target_label))
    # if this is a complete match
    if count_num_same == len(target_label):
      count_complete_match += 1

    print("label: {} ## t:{} vs o:{} ## with loss: {}".format(ll, target_label, outputs, one_ppx))
    print("--------------------------")
    if check_order(target_label, outputs):
      count_right_order += 1

  print("Overall parts matched per command in precentage: {} in {}".format(np.average(count_portion_match), set_name))
  print("precentage of correct ordered command: {} in {}".format(count_right_order*1.0/size, set_name))
  print("Number of complete matches: {} out of {} in {}".format(count_complete_match, size, set_name))


def decode():
  with tf.Session() as sess:
    # Create model and load parameters.
    model = create_model(sess, True)
    model.batch_size = 1  # We decode one sentence at a time.

    # Load vocabularies.
    from_vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab%d.from" % FLAGS.from_vocab_size)
    to_vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab%d.to" % FLAGS.to_vocab_size)
    from_vocab, _ = data_utils.initialize_vocabulary(from_vocab_path)

    # A reverse dictory for integer encoding of commands
    to_vocab, rev_to_vocab = data_utils.initialize_vocabulary(to_vocab_path)

    # Decode from standard input.
    if FLAGS.self_test:
      sys.stdout.write("> ")
      sys.stdout.flush()
      sentence = sys.stdin.readline()
      while sentence:
        # Get token-ids for the input sentence.
        token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), from_vocab)
        # Which bucket does it belong to?
        bucket_id = len(_buckets) - 1
        for i, bucket in enumerate(_buckets):
          if bucket[0] >= len(token_ids):
            bucket_id = i
            break
        else:
          logging.warning("Sentence truncated: %s", sentence)

        # Get a 1-element batch to feed the sentence to the model.
        encoder_inputs, decoder_inputs, target_weights = model.get_batch(
            {bucket_id: [(token_ids, [])]}, bucket_id)
        # Get output logits for the sentence.
        _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                         target_weights, bucket_id, True)
        # This is a greedy decoder - outputs are just argmaxes of output_logits.
        outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
        #print("output: {}".format(outputs[0]))
        # If there is an EOS symbol in outputs, cut them at that point.
        if data_utils.EOS_ID in outputs:
          outputs = outputs[:outputs.index(data_utils.EOS_ID)]
        # Print out French sentence corresponding to outputs.
        print(" ".join([tf.compat.as_str(rev_to_vocab[output]) for output in outputs]))
        print("> ", end="")
        sys.stdout.flush()
        sentence = sys.stdin.readline()
    else:
      # Read both the training and validation data/label and evaluation metrics
      # 1. How many are exactly correct
      # 2. How many parts of a command is correct on average
      # Evaluate on training set.
      #with open('data/data.txt', 'r') as data, open('data/label.txt', 'r') as label:
      #  decode_helper(data, label, 5030, 'Training Set', from_vocab, to_vocab, model, sess)
      with open('data/validation/data.txt', 'r') as data, open('data/validation/label.txt', 'r') as label:
        decode_helper(data, label, 298, 'Validation Set', from_vocab, to_vocab, model, sess)

def main(_):
  if FLAGS.decode:
    decode()
  else:
    train()

if __name__ == "__main__":
  tf.app.run()

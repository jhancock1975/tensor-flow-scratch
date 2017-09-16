# copied from youtube http://www.youtube.com/watch?=PicxU81owCs
# Sandford lecture on tensorflow lecture 7: introudction to tensorflow

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import math

from utils import *

batch_size=128

vocabulary_size = 50000

embedding_size = 128

num_sampled = 64

train_data, val_data, reverse_dictionary = load_data()
print("number of training examples: ", len(train_data)*batch_size)
print("Number of validation examples: ", len(val_data))

def skipgram():

  batch_inputs  = tf.placeholder(tf.int32, shape=[batch_size,])
  batch_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
  val_dataset = tf.constant(val_data, dtype=tf.int32)

  with tf.variable_scope("word2vec") as scope:
      embeddings = tf.Variable(tf.random_uniform([vocabular_size, 
                                                  embedding_size],
                                                  -1.0, 1.0))
      batch_embeddings = tf.nn.embedding_lookup(embeddings, batch_inputs)

      weights = tf.Variable(tf.truncated_normal([vocabulary_size, 
                                                 embedding_size]),
                                                 stddev=1.0/math.sqrt(embedding_size))

      biases = tf.Variable(tf.zeros[vocabulary_size])
      loss = tf.reduce_mean(tf.nn.nce_loss(wieghts,
					    biases=biases
					    labels=batch_labels
					    inputs=batch_inputs
					    num_sampled=num_sampled
					    num_classes=vocabulary_size))

       norm = tf.sqrt(tf.reducce_mean(tf.square(embeddings), 1, keep_dims=True))
       normalized_embeddings = embeddings/norm

	similarity = tf.matmul(val_embeddings, normalized_embeddings, transpose_b=True)

       val_embeddings = tf.nn.embedding_lookup(normalized_embeddings, val_dataset)

       return batch_inputs, batch_labels, normalized_embeddings, loss, similarity 

def run():
  batch_inputs, batch_labels, normalized_embeddings, loss, similarity = skipgram()

  optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

  init = tf.global_variables_initializer()

  with tf.Session() as sess:
    sess.run(init)

    average_loss = 0.00
    for step, batch_data in enumerate(train_data):
      inputs, labels = batch_data
      feed_dict = {batch_inputs: inputs, batch_labels: labels}
      _, loss_val = sess.run([optimizer, loss], feed_dict)
      average_loss += loss_val
      if step % 1000 == 0:
        if step > 0:
          average_loss /= 1000 
          print ('loss at iteration: ', step,':', average_loss)
          average_loss = 0


       if step % 5000 == 0:
         sim = similarity.eval()
           for i in xrange(len(val_data)):
             top_k  = 8
             nearest = (sim[i, :]).argsort()[1:top_k+1]       
             print_closest_words(val_data[i], nearest, reverse_dictionary)

      final_embeddings = normalized_embedding.eval()
  writer=tf.summary.FileWriter('./graph', session.graph)
  writer.close()


final_embeddings = run()

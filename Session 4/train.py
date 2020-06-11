# -*- coding: utf-8 -*-

from data_reader import *
from RNN import *
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

def train_and_evaluate_RNN():
    with open('../datasets/w2v/vocab-raw.txt') as f:
        vocab_size = len(f.read().splitlines())

    tf.set_random_seed(2020)
    rnn = RNN(
        vocab_size=vocab_size,
        embedding_size= 300,
        lstm_size= 50,
        batch_size= 50
    )

    predicted_labels, loss = rnn.build_graph()
    train_op = rnn.trainer(loss=loss, learning_rate=0.01)

    with tf.Session() as sess:
        train_data_reader = DataReader(
            data_path= '../datasets/w2v/20news-train-encoded.txt',
            batch_size=50
        )

        test_data_reader = DataReader(
                data_path= '../datasets/w2v/20news-test-encoded.txt',
                batch_size=50
        )

        step = 0
        MAX_STEP = 1000

        sess.run(tf.global_variables_initializer())

        while step < MAX_STEP:
            next_train_batch = train_data_reader.next_batch()
            train_data, train_labels, train_sentence_lengths = next_train_batch
            plabels_eval, loss_eval, _ = sess.run(
                [predicted_labels, loss, train_op],
                feed_dict={
                    rnn._data: train_data,
                    rnn._labels: train_labels,
                    rnn._sentence_lengths: train_sentence_lengths,
                    #rnn._final_tokens: train_final_tokens
                }
            )
            step += 1
            if step % 20 == 0:
                print('loss: ', loss_eval)

            # danh gia tren test data
            if train_data_reader._current_part == 0:
                num_true_preds = 0

                while True:
                    next_test_batch = test_data_reader.next_batch()
                    test_data, test_labels, test_sentence_lengths = next_test_batch

                    test_plabels_eval = sess.run(
                        predicted_labels,
                        feed_dict={
                            rnn._data: test_data,
                            rnn._labels: test_labels,
                            rnn._sentence_lengths: test_sentence_lengths,
                            #rnn._final_tokens: test_final_tokens
                        }
                    )

                    matches = np.equal(test_plabels_eval, test_labels)
                    num_true_preds += np.sum(matches.astype(float))

                    if test_data_reader._current_part == 0:
                        break

                print("Epoch: ", train_data_reader._num_epoch)
                print("Accuracy on test data: ", num_true_preds * 100. / len(test_data_reader._data))

if __name__ == "__main__":
    tf.reset_default_graph()
    train_and_evaluate_RNN()


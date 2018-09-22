#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import sys
sys.path.append('.')
from utils import dataset as data


class Network:
    def __init__(self, threads, seed=541):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(
            graph=graph,
            config=tf.ConfigProto(
                inter_op_parallelism_threads=threads,
                intra_op_parallelism_threads=threads))

    def construct(self, args, num_tokens, num_labels):
        with self.session.graph.as_default():
            # Inputs
            self.tokens_ids = tf.placeholder(
                tf.int32, [None, None], name="tokens_ids")
            self.terms_lens = tf.placeholder(
                tf.int32, [None], name="terms_lens")
            self.labels = tf.placeholder(tf.int32, [None], name="labels")

            # Token embeddings
            self.token_embeddings = tf.get_variable(
                "token_embeddings",
                shape=[num_tokens, args.embed_dim],
                dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(self.token_embeddings,
                                              self.tokens_ids)

            # Computation
            # rnn_cell = tf.nn.rnn_cell.BasicLSTMCell
            rnn_cell = tf.nn.rnn_cell.GRUCell

            with tf.variable_scope('bi_rnn'):
                _, (state_fwd, state_bwd) = \
                        tf.nn.bidirectional_dynamic_rnn(
                            rnn_cell(args.rnn_cell_dim),
                            rnn_cell(args.rnn_cell_dim),
                            inputs,
                            sequence_length=self.terms_lens,
                            dtype=tf.float32)

            state = tf.concat([state_fwd, state_bwd], axis=-1)
            #print('state shape :', state.get_shape())
            layers = [state] # TODO is it OK?
            for _ in range(args.num_dense_layers):
                layers.append(
                    tf.layers.dense(layers[-1],
                                    args.dense_layer_units,
                                    activation=tf.nn.relu)
                )
            self.logits = tf.layers.dense(layers[-1], num_labels, name='logits')
            self.logits = tf.nn.softmax(self.logits)
            self.logits_0 = self.logits[:0] # TODO nicer
            self.predictions = tf.argmax(self.logits, axis=1, name='predictions')
            #predictions_shape = tf.shape(self.predictions)
            #print('self.predictions shape: ', self.predictions.get_shape())

            # Training
            loss = tf.losses.sparse_softmax_cross_entropy(
                self.labels, self.logits)
            global_step = tf.train.create_global_step()
            self.training = tf.train.AdamOptimizer().minimize(
                loss, global_step=global_step, name="training")

            # Summaries
            self.current_accuracy, self.update_accuracy = tf.metrics.accuracy(
                self.labels, self.predictions)
            self.current_loss, self.update_loss = tf.metrics.mean(loss)
            self.reset_metrics = tf.variables_initializer(
                tf.get_collection(tf.GraphKeys.METRIC_VARIABLES))

            summary_writer = tf.contrib.summary.create_file_writer(
                args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), \
                    tf.contrib.summary.record_summaries_every_n_global_steps(10):
                self.summaries['train'] = [
                    tf.contrib.summary.scalar(
                        'train/loss',
                        self.update_loss),
                    tf.contrib.summary.scalar(
                        'train/accuracy',
                        self.update_accuracy)]
            with summary_writer.as_default(), \
                    tf.contrib.summary.always_record_summaries():
                self.summaries['valid'] = [
                    tf.contrib.summary.scalar(
                        'valid/loss',
                        self.current_loss),
                    tf.contrib.summary.scalar(
                        'valid/accuracy',
                        self.current_accuracy)]

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(
                    session=self.session, graph=self.session.graph)

            # Saver
            tf.add_to_collection('end_points/tokens_ids',
                                 self.tokens_ids)
            tf.add_to_collection('end_points/terms_lens',
                                 self.terms_lens)
            tf.add_to_collection('end_points/predictions',
                                 self.predictions)
            tf.add_to_collection('end_points/logits',
                                 self.logits)

            self.saver = tf.train.Saver()

    def save(self, path):
        return self.saver.save(self.session, path)

    def train_epoch(self, train_set, batch_size):
        while not train_set.epoch_finished():
            tokens_ids, terms_lens, labels = train_set.next_batch(batch_size)
            self.session.run(self.reset_metrics)
            self.session.run([self.training, self.summaries["train"]],
                             {self.terms_lens: terms_lens,
                              self.tokens_ids: tokens_ids,
                              self.labels: labels})

    def train_batch(self, batch):
        tokens_ids, terms_lens, labels = batch
        self.session.run(self.reset_metrics)
        self.session.run([self.training, self.summaries["train"]],
                         {self.terms_lens: terms_lens,
                          self.tokens_ids: tokens_ids,
                          self.labels: labels})

    def evaluate(self, dataset_name, dataset, batch_size):
        self.session.run(self.reset_metrics)
        while not dataset.epoch_finished():
            tokens_ids, terms_lens, labels = dataset.next_batch(batch_size)
            self.session.run([self.update_accuracy, self.update_loss],
                             {self.terms_lens: terms_lens,
                              self.tokens_ids: tokens_ids,
                              self.labels: labels})
        return self.session.run(
            [self.current_accuracy, self.summaries[dataset_name]])[0]

    def embeddings(self):
        return self.session.run(self.token_embeddings)

class NetworkPredict:
    def __init__(self, threads=1, seed=541):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(
            graph=graph,
            config=tf.ConfigProto(
                inter_op_parallelism_threads=threads,
                intra_op_parallelism_threads=threads))

    def load(self, path):
        # Load the metagraph
        with self.session.graph.as_default():
            self.saver = tf.train.import_meta_graph(path + '.meta')

            # Attach the end points
            self.tokens_ids = tf.get_collection(
                'end_points/tokens_ids')[0]
            self.terms_lens = tf.get_collection(
                'end_points/terms_lens')[0]
            self.predictions = tf.get_collection(
                'end_points/predictions')[0]

        # Load the graph weights
        self.saver.restore(self.session, path)

    def predict(self, dataset_name, dataset):
        tokens_ids, terms_lens, = dataset.test()
        return self.session.run(self.predictions,
                         {self.terms_lens: terms_lens,
                          self.tokens_ids: tokens_ids})

if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(541)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vocab",
        type=str,
        help="Path to a vocabulary file.")
    parser.add_argument(
        "--train_set",
        default='data/split/equiv.train',
        type=str,
        help="Path to a training set.")
    parser.add_argument(
        "--valid_set",
        default='data/split/equiv.valid',
        type=str,
        help="Path to a validation set.")
    parser.add_argument(
        "--test_set",
        default='',
        type=str,
        help="Path to a testing set.")
    parser.add_argument(
        "--model_path",
        default='',
        type=str,
        help="Path where to save the trained model.")
    parser.add_argument(
        "--batch_size",
        default=8,
        type=int,
        help="Batch size.")
    parser.add_argument(
        "--epochs",
        default=8,
        type=int,
        help="Number of epochs.")
    parser.add_argument(
        "--embed_dim",
        default=8,
        type=int,
        help="Token embedding dimension.")
    parser.add_argument(
        "--rnn_cell_dim",
        default=8,
        type=int,
        help="RNN cell dimension.")
    parser.add_argument(
        "--num_dense_layers",
        default=2,
        type=int,
        help="Number of dense layers.")
    parser.add_argument(
        "--dense_layer_units",
        default=8,
        type=int,
        help="Number of units in each dense layer.")
    parser.add_argument(
        "--threads",
        default=1,
        type=int,
        help="Maximum number of threads to use.")
    parser.add_argument(
        "--logdir",
        default='',
        type=str,
        help="Logdir.")
    args = parser.parse_args()

    if args.model_path:
        args.logdir = args.model_path
    else:
        if not args.logdir:
            # Create dir for logs
            if not os.path.exists("logs"):
                os.mkdir("logs")

            # Create logdir name
            args.logdir = "logs/{}--{}--{}".format(
                os.path.basename(__file__),
                datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S"),
                ",".join(
                    ("{}={}".format(
                        re.sub("(.)[^_]*_?", r"\1", key), value) \
                            for key, value in sorted(vars(args).items()) \
                                if not '/' in str(value) \
                                and not 'threads' in key
                                and not 'logdir' in key
                    )
                )
            )

    print("The logdir is: {}".format(args.logdir))

    # Load the data
    train_set = data.Dataset(args.train_set, args.vocab, shuffle_batches=True)
    valid_set = data.Dataset(args.valid_set, args.vocab, shuffle_batches=False)

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args, train_set.num_tokens, train_set.num_labels)

    # Train, batches
    print("Training started.")
    for i in range(args.epochs):
        while not train_set.epoch_finished():
            batch = train_set.next_batch(args.batch_size)
            network.train_batch(batch)

            # Saving embeddings
            #embeddings = network.embeddings()
            #time = datetime.datetime.now().strftime("%H%M%S")
            #file_name = args.logdir + '/embeddings_' + time + '.csv'
            #embeddings_to_write = '\n'.join(
            #    [','.join([str(round(j, 6)) for j in i]) for i in embeddings])
            #with open(file_name, 'w') as f:
            #    f.write(embeddings_to_write + '\n')
        accuracy = network.evaluate('valid', valid_set, args.batch_size)
        print("Accuracy on valid set after epoch {}: {:.2f}".format(
                                            i + 1, 100 * accuracy))
    print("Training finished.")

    # Save model
    model_path = network.save(args.logdir + '/model')
    print('Saved model path: ', model_path)

    if args.test_set:
        network = NetworkPredict()
        network.load(model_path)
        test = data.Dataset(args.test_set, args.vocab, test=True)
        p = network.predict('test', test)
        for i in p[:10]:
            print(i)



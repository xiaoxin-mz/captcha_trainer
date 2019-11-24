#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import tensorflow as tf
import core
import utils
import utils.data
import validation
from config import *
from tf_graph_util import convert_variables_to_constants
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)


class Trains:

    def __init__(self, model_conf: ModelConfig):

        self.model_conf = model_conf
        self.validation = validation.Validation(self.model_conf)

    def compile_graph(self, acc):
        input_graph = tf.Graph()
        predict_sess = tf.Session(graph=input_graph)

        with predict_sess.graph.as_default():
            model = core.NeuralNetwork(
                model_conf=self.model_conf,
                mode=RunMode.Predict,
                cnn=NETWORK_MAP[self.model_conf.neu_cnn],
                recurrent=NETWORK_MAP[self.model_conf.neu_recurrent]
            )
            model.build_graph()
            input_graph_def = predict_sess.graph.as_graph_def()
            saver = tf.train.Saver(var_list=tf.global_variables())
            tf.logging.info(tf.train.latest_checkpoint(self.model_conf.model_root_path))
            saver.restore(predict_sess, tf.train.latest_checkpoint(self.model_conf.model_root_path))
            tf.keras.backend.set_session(session=predict_sess)

            output_graph_def = convert_variables_to_constants(
                predict_sess,
                input_graph_def,
                output_node_names=['dense_decoded']
            )

        if not os.path.exists(self.model_conf.compile_model_path):
            os.makedirs(self.model_conf.compile_model_path)

        last_compile_model_path = (
            os.path.join(self.model_conf.compile_model_path, "{}.pb".format(self.model_conf.model_name))
        ).replace('.pb', '_{}.pb'.format(int(acc * 10000)))

        with tf.io.gfile.GFile(last_compile_model_path, mode='wb') as gf:
            gf.write(output_graph_def.SerializeToString())

        self.generate_config(acc)

    def train_process(self):

        self.model_conf.println()

        model = core.NeuralNetwork(
            model_conf=self.model_conf,
            mode=RunMode.Trains,
            cnn=self.model_conf.neu_cnn,
            recurrent=NETWORK_MAP[self.model_conf.neu_recurrent] if self.model_conf.neu_recurrent else None
        )
        model.build_graph()

        tf.compat.v1.logging.info('Loading Trains DataSet...')
        train_feeder = utils.data.DataIterator(model_conf=self.model_conf, mode=RunMode.Trains)
        train_feeder.read_sample_from_tfrecords(self.model_conf.trains_path)
        tf.compat.v1.logging.info('Loading Test DataSet...')
        validation_feeder = utils.data.DataIterator(model_conf=self.model_conf, mode=RunMode.Validation)
        validation_feeder.read_sample_from_tfrecords(self.model_conf.validation_path)

        tf.logging.info('Total {} Trains DataSets'.format(train_feeder.size))
        tf.logging.info('Total {} Test DataSets'.format(validation_feeder.size))
        if validation_feeder.size >= train_feeder.size:
            exception("The number of training sets cannot be less than the test set.", )

        num_train_samples = train_feeder.size
        num_test_samples = validation_feeder.size
        if num_test_samples < self.model_conf.validation_batch_size:
            exception(
                "The number of test sets cannot be less than the test batch size.",
                ConfigException.INSUFFICIENT_SAMPLE
            )
        num_batches_per_epoch = int(num_train_samples / self.model_conf.batch_size)

        sess_config = tf.compat.v1.ConfigProto(
            # allow_soft_placement=True,
            log_device_placement=False,
            gpu_options=tf.compat.v1.GPUOptions(
                allocator_type='BFC',
                allow_growth=True,  # it will cause fragmentation.
                per_process_gpu_memory_fraction=self.model_conf.memory_usage)
        )
        accuracy = 0
        epoch_count = 1
        with tf.compat.v1.Session(config=sess_config) as sess:
            tf.keras.backend.set_session(session=sess)
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=2)
            train_writer = tf.compat.v1.summary.FileWriter('logs', sess.graph)
            # try:
            checkpoint_state = tf.train.get_checkpoint_state(self.model_conf.model_root_path)
            if checkpoint_state and checkpoint_state.model_checkpoint_path:
                saver.restore(sess, checkpoint_state.model_checkpoint_path)

            tf.logging.info('Start training...')

            while 1:
                start_time = time.time()
                epoch_cost = 0
                for cur_batch in range(num_batches_per_epoch):
                    batch_time = time.time()

                    batch = train_feeder.generate_batch_by_tfrecords(sess)

                    batch_inputs, batch_labels = batch

                    feed = {
                        model.inputs: batch_inputs,
                        model.labels: batch_labels,
                    }

                    summary_str, batch_cost, step, _, seq_len = sess.run(
                        [model.merged_summary, model.cost, model.global_step, model.train_op, model.seq_len],
                        feed_dict=feed
                    )
                    train_writer.add_summary(summary_str, step)
                    if step % 100 == 0 and step != 0:
                        tf.logging.info(
                            'Step: {} Time: {:.3f} sec/batch, Cost = {:.8f}, BatchSize: {}, RNNTimeStep: {}'.format(
                                step,
                                time.time() - batch_time,
                                batch_cost,
                                len(batch_inputs),
                                seq_len[0]
                            )
                        )
                    if step % self.model_conf.trains_save_steps == 0 and step != 0:
                        saver.save(sess, self.model_conf.save_model, global_step=step)

                    if step % self.model_conf.trains_validation_steps == 0 and step != 0:

                        batch_time = time.time()
                        batch = validation_feeder.generate_batch_by_tfrecords(sess)

                        test_inputs, test_labels = batch
                        val_feed = {
                            model.inputs: test_inputs,
                            model.labels: test_labels
                        }
                        dense_decoded, lr = sess.run(
                            [model.dense_decoded, model.lrn_rate],
                            feed_dict=val_feed
                        )

                        accuracy = self.validation.accuracy_calculation(
                            validation_feeder.labels,
                            dense_decoded,
                        )
                        log = "Epoch: {}, Step: {}, Accuracy = {:.4f}, Cost = {:.5f}, " \
                              "Time = {:.3f} sec/batch, LearningRate: {}"
                        tf.logging.info(log.format(
                            epoch_count,
                            step,
                            accuracy,
                            batch_cost,
                            time.time() - batch_time,
                            lr / len(batch),
                        ))
                        # epoch_cost = batch_cost
                        achieve_accuracy = accuracy >= self.model_conf.trains_end_acc
                        achieve_epochs = epoch_count >= self.model_conf.trains_end_epochs
                        achieve_cost = batch_cost <= self.model_conf.trains_end_cost
                        over_epochs = epoch_count > 10000

                        if (achieve_accuracy and achieve_epochs and achieve_cost) or over_epochs:
                            break

                achieve_accuracy = accuracy >= self.model_conf.trains_end_acc
                achieve_epochs = epoch_count >= self.model_conf.trains_end_epochs
                achieve_cost = batch_cost <= self.model_conf.trains_end_cost
                over_epochs = epoch_count > 10000
                if (achieve_accuracy and achieve_epochs and achieve_cost) or over_epochs:
                    self.compile_graph(accuracy)
                    tf.logging.info('Total Time: {} sec.'.format(time.time() - start_time))
                    break
                epoch_count += 1

    def generate_config(self, acc):
        with open(self.model_conf.model_conf_path, "r", encoding="utf8") as current_fp:
            text = "".join(current_fp.readlines())
            text = text.replace("ModelName: {}".format(self.model_conf.model_name),
                                "ModelName: {}_{}".format(self.model_conf.model_name, int(acc * 10000)))
        compiled_config_path = os.path.join(self.model_conf.output_path, "{}/model".format(self.model_conf.model_name))

        if not os.path.exists(compiled_config_path):
            os.makedirs(compiled_config_path)

        compiled_config_path = os.path.join(compiled_config_path, "{}_model.yaml".format(self.model_conf.model_name))
        with open(compiled_config_path, "w", encoding="utf8") as save_fp:
            save_fp.write(text)


def main(argv):
    project_name = argv[-1]
    model_conf = ModelConfig(project_name=project_name)
    Trains(model_conf).train_process()
    tf.logging.info('Training completed.')
    pass


if __name__ == '__main__':

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    tf.compat.v1.app.run()


#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import time
import numpy as np
import tensorflow as tf
import framework
import utils
from config import *
from tf_graph_util import convert_variables_to_constants
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)


def compile_graph(acc):
    input_graph = tf.Graph()
    predict_sess = tf.Session(graph=input_graph)

    with predict_sess.graph.as_default():
        model = framework.GraphOCR(
            RunMode.Predict,
            NETWORK_MAP[NEU_CNN],
            NETWORK_MAP[NEU_RECURRENT]
        )
        model.build_graph()
        input_graph_def = predict_sess.graph.as_graph_def()
        saver = tf.train.Saver(var_list=tf.global_variables())
        tf.logging.info(tf.train.latest_checkpoint(MODEL_PATH))
        saver.restore(predict_sess, tf.train.latest_checkpoint(MODEL_PATH))
        tf.keras.backend.set_session(session=predict_sess)

        output_graph_def = convert_variables_to_constants(
            predict_sess,
            input_graph_def,
            output_node_names=['dense_decoded']
        )

    if not os.path.exists(COMPILE_MODEL_PATH):
        os.makedirs(COMPILE_MODEL_PATH)

    last_compile_model_path = (
        os.path.join(COMPILE_MODEL_PATH, "{}.pb".format(TARGET_MODEL))
    ).replace('.pb', '_{}.pb'.format(int(acc * 10000)))

    with tf.io.gfile.GFile(last_compile_model_path, mode='wb') as gf:
        gf.write(output_graph_def.SerializeToString())

    generate_config(acc)


def train_process(mode=RunMode.Trains):
    model = framework.GraphOCR(mode, NETWORK_MAP[NEU_CNN], NETWORK_MAP[NEU_RECURRENT])
    model.build_graph()

    tf.compat.v1.logging.info('Loading Trains DataSet...')
    train_feeder = utils.DataIterator(mode=RunMode.Trains)
    train_feeder.read_sample_from_tfrecords(TRAINS_PATH)
    tf.compat.v1.logging.info('Loading Test DataSet...')
    test_feeder = utils.DataIterator(mode=RunMode.Test)
    test_feeder.read_sample_from_tfrecords(TEST_PATH)

    tf.logging.info('Total {} Trains DataSets'.format(train_feeder.size))
    tf.logging.info('Total {} Test DataSets'.format(test_feeder.size))
    if test_feeder.size >= train_feeder.size:
        exception("The number of training sets cannot be less than the test set.", )

    num_train_samples = train_feeder.size
    num_test_samples = test_feeder.size
    if num_test_samples < TEST_BATCH_SIZE:
        exception(
            "The number of test sets cannot be less than the test batch size.",
            ConfigException.INSUFFICIENT_SAMPLE
        )
    num_batches_per_epoch = int(num_train_samples / BATCH_SIZE)

    config = tf.compat.v1.ConfigProto(
        # allow_soft_placement=True,
        log_device_placement=False,
        gpu_options=tf.compat.v1.GPUOptions(
            allocator_type='BFC',
            allow_growth=True,  # it will cause fragmentation.
            per_process_gpu_memory_fraction=GPU_USAGE)
    )
    accuracy = 0
    epoch_count = 1
    with tf.compat.v1.Session(config=config) as sess:
        tf.keras.backend.set_session(session=sess)
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=2)
        train_writer = tf.compat.v1.summary.FileWriter('logs', sess.graph)
        # try:
        checkpoint_state = tf.train.get_checkpoint_state(MODEL_PATH)
        if checkpoint_state and checkpoint_state.model_checkpoint_path:
            saver.restore(sess, checkpoint_state.model_checkpoint_path)

        tf.logging.info('Start training...')

        while 1:
            start_time = time.time()
            epoch_cost = 0
            for cur_batch in range(num_batches_per_epoch):
                batch_time = time.time()

                batch = train_feeder.generate_batch_by_tfrecords(sess)

                batch_inputs, batch_seq_len, batch_labels = batch
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
                if step % TRAINS_SAVE_STEPS == 0 and step != 0:
                    saver.save(sess, SAVE_MODEL, global_step=step)

                if step % TRAINS_VALIDATION_STEPS == 0 and step != 0:

                    batch_time = time.time()
                    batch = test_feeder.generate_batch_by_tfrecords(sess)

                    test_inputs, batch_seq_len, test_labels = batch
                    val_feed = {
                        model.inputs: test_inputs,
                        model.labels: test_labels
                    }
                    dense_decoded, lr = sess.run(
                        [model.dense_decoded, model.lrn_rate],
                        feed_dict=val_feed
                    )

                    accuracy = utils.accuracy_calculation(
                        test_feeder.labels,
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
                    epoch_cost = batch_cost
                    if accuracy >= TRAINS_END_ACC and epoch_count >= TRAINS_END_EPOCHS and batch_cost <= TRAINS_END_COST or epoch_count > 10000:
                        break

            if accuracy >= TRAINS_END_ACC and epoch_count >= TRAINS_END_EPOCHS and epoch_cost <= TRAINS_END_COST or epoch_count > 10000:
                compile_graph(accuracy)
                tf.logging.info('Total Time: {} sec.'.format(time.time() - start_time))
                break
            epoch_count += 1


def generate_config(acc):
    with open(MODEL_CONFIG_PATH, "r", encoding="utf8") as current_fp:
        text = "".join(current_fp.readlines())
        text = text.replace("ModelName: {}".format(TARGET_MODEL),
                            "ModelName: {}_{}".format(TARGET_MODEL, int(acc * 10000)))
    compiled_config_path = os.path.join(OUTPUT_PATH, "{}/model".format(TARGET_MODEL))

    if not os.path.exists(compiled_config_path):
        os.makedirs(compiled_config_path)

    compiled_config_path = os.path.join(compiled_config_path, "{}_model.yaml".format(TARGET_MODEL))
    with open(compiled_config_path, "w", encoding="utf8") as save_fp:
        save_fp.write(text)


def init_dataset_config():

    with open(MODEL_CONFIG_PATH, "r", encoding="utf8") as current_fp:
        text = "".join(current_fp.readlines())
        output_filename = lambda value: os.path.join(TFRECORDS_DIR, "{}_{}.tfrecords".format(TARGET_MODEL, value))
        target_test = re.search(" {2}TestPath:.*\n", text)
        target_test = target_test.group(0) if target_test else None
        target_trains = re.search(" {2}TrainsPath:.*\n", text)
        target_trains = target_trains.group(0) if target_trains else None

        TRAINS_PATH = output_filename('trains')
        TEST_PATH = output_filename('test')

        if not target_trains:
            text = text.replace("\n  DatasetPath", "\n  TrainsPath: {}\n  DatasetPath".format(output_filename('trains')))
        else:
            text = text.replace(target_trains, "  TrainsPath: {}\n".format(output_filename('trains')))
        if not target_test:
            text = text.replace("\n  DatasetPath", "\n  TestPath: {}\n  DatasetPath".format(output_filename('test')))
        else:
            text = text.replace(target_test, "  TestPath: {}\n".format(output_filename('test')))

    with open(MODEL_CONFIG_PATH, "w", encoding="utf8") as save_fp:
        save_fp.write(text)


def main(_):
    if not TRAINS_PATH or not TEST_PATH:
        init_dataset_config()
        from make_dataset import make_dataset
        make_dataset()

    init()
    train_process()
    tf.logging.info('Training completed.')
    pass


if __name__ == '__main__':
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    tf.compat.v1.app.run()

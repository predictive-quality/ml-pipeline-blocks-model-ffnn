# Copyright (c) 2022 RWTH Aachen - Werkzeugmaschinenlabor (WZL)
# Contact: Simon Cramer, s.cramer@wzl-mq.rwth-aachen.de

from absl import logging, flags, app
import os
from Ffnn import Ffnn
import pandas as pd
import numpy as np
import json
import tensorflow as tf
from s3_smart_open import *
from tensorflow.python.client import device_lib

flags.DEFINE_string('input_path',None,'Path where training dataset is found.')
flags.DEFINE_string('output_path',None,'Path where to store model (if save_model is true).')
flags.DEFINE_string('model_input_path', '', 'Path to load model from to continue training.')
flags.DEFINE_string('model_name',None,'Name of the model to save/load.')
flags.DEFINE_enum('stage', None, ['fit', 'predict','evaluate'], 'Job status.')
flags.DEFINE_list('layers', '', 'Define the layers, cmp. tf.keras.layers.Dense.get_config() ')
flags.DEFINE_string('loss','MeanAbsoluteError', 'Loss function for NN. See here for candidate functions: https://www.tensorflow.org/api_docs/python/tf/keras/losses')
flags.DEFINE_list('metrics', '', 'Metrics for on training evaluation. See here for candidate functions: https://www.tensorflow.org/api_docs/python/tf/keras/metrics')
flags.DEFINE_string('optimizer_config',"{'class_name': 'sgd', 'config':{}}", 'Dict providing info about optimizer')
flags.DEFINE_string('callbacks', '{}', 'Callbacks for training. Example: {"TensorBoard": {"log_dir": "./logs"}}. Path is relative to output_path.  See here for candidates: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks')
flags.DEFINE_integer('batch_size',32,'Batch size for fit and predict')
flags.DEFINE_integer('epochs',100,'Epochs for fit')
flags.DEFINE_integer('verbose',2,'Choice of 0,1,2. See  https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit')
flags.DEFINE_integer('validation_freq',1,'How often validation is done. https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit')
flags.DEFINE_float('validation_split',0.1,'Percentage of dataset for validation purposes. https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit')
flags.DEFINE_integer('buffer_size',100,'Size of buffer for shuffling.')
flags.DEFINE_string('CUDA_VISIBLE_DEVICES',None,'Limit TF to only certain gpu.')
flags.DEFINE_list('columns_y',None,'Name of the y column to train,predict')
flags.DEFINE_list('set_gpus',None,'Set the numbers of the Gpu')
flags.DEFINE_string('filename_x',None,'Filename of _x.fth file')
flags.DEFINE_string('filename_y',None,'Filename of _y.fth file')


FLAGS = flags.FLAGS

flags.mark_flag_as_required('input_path')
flags.mark_flag_as_required('output_path')
flags.mark_flag_as_required('model_name')
flags.mark_flag_as_required('stage')
flags.mark_flag_as_required('filename_x')

def main(argv):
    """Runs three possible stages on a FeedForward Neural Network.
    Required flags:
        input_path
        output_path
        model_name
        stage
        filename_x

        1. Fit
          Train or fit the neural network to a given dataset.
          Required Flags:
            units
          Required input data:
            input_path/train_x.fth
            input_path/train_y.fth
          Output data:
            input_path/model_name
      2. Predict
          Get predictions from the FFNN given an input. Model is loaded from disk.
          Required input data:
              input_path/predict_x.fth
              input_path/model_name
          Output data:
              output_path/results.fth
      3. Evaluate
          Score the performance of the estimator against a given set of metrics.
          The model is loaded from disk. Results are provided in a dictonary with the format metric_name: metric[ndarray]
              Required input data:
                  input_path/evaluate_x.fth
                  input_path/evaluate_y.fth
                  input_path/model_name
              Output data:
                output_path/metrics.json

        Args:
            argv (None): No further arguments should be parsed.

        Raises:
            ValueError: If input path is not existant.
    """
    if not FLAGS.set_gpus == None: 
        gpus = tf.config.list_physical_devices('GPU')
        gpus_before = len(gpus)
        assert len(gpus) >= len(FLAGS.set_gpus), 'More GPUS set than given!'
        if gpus:
            # Restrict TensorFlow to only use the first GPU
            set_gpus = [int(i) for i in FLAGS.set_gpus]
            for i in range(len(gpus)-1,-1,-1):
                if i not in set_gpus:
                    gpus.remove(gpus[i])
            try:
                tf.config.set_visible_devices(gpus[:], 'GPU')
            except RuntimeError as e:
                # Visible devices must be set before GPUs have been initialized
                logging.warning(e)
            logical_gpus = tf.config.list_logical_devices('GPU')
            logging.info("Physical GPUs: {} Logical GPU: {}".format(gpus_before,len(logical_gpus)))


    to_txt(FLAGS.output_path,'flagfile_'+FLAGS.stage+'.txt',FLAGS.flags_into_string())

    if FLAGS.stage == 'fit':
        # Load Data
        X=read_pd_fth(FLAGS.input_path,FLAGS.filename_x)
        y = Ffnn.read_y(FLAGS.input_path,FLAGS.filename_y, FLAGS.columns_y)
        # Reformat flags
        layers = [eval(ffnn_layer) for ffnn_layer in FLAGS.layers]
        optimizer_config = eval(FLAGS.optimizer_config)
        callbacks = eval(FLAGS.callbacks)

        # Make Tensorboard path relative to output_path
        if 'TensorBoard' in callbacks.keys():
            if not FLAGS.output_path[:5]  == 's3://':
                callbacks['TensorBoard']['log_dir'] = os.path.join(FLAGS.output_path,callbacks['TensorBoard']['log_dir'])
            else:
                callbacks['TensorBoard']['log_dir'] = os.path.join(os.getcwd(),callbacks['TensorBoard']['log_dir'])
                
        logging.warning(callbacks['TensorBoard']['log_dir'])
        ffnn = Ffnn(input_shape=(X.values.shape[1],),
                    layers=layers,
                    loss=FLAGS.loss,
                    metrics=FLAGS.metrics,
                    optimizer_config=optimizer_config,
                    callbacks=callbacks,
                    batch_size=FLAGS.batch_size,
                    epochs=FLAGS.epochs,
                    verbose=FLAGS.verbose,
                    validation_freq=FLAGS.validation_freq,
                    validation_split=FLAGS.validation_split,
                    buffer_size=FLAGS.buffer_size,
                    model_input_path=FLAGS.model_input_path,
                    model_name=FLAGS.model_name)

        ffnn.fit(X,y)

        # Save Object
        ffnn.save(FLAGS.output_path, FLAGS.model_name)
        if FLAGS.output_path[:5]  == 's3://':
            ffnn.del_locals(FLAGS.model_name,FLAGS.stage)
        return


    ffnn = Ffnn.load(FLAGS.input_path, FLAGS.model_name)
    if FLAGS.stage == 'predict':
        # Load Data
        X=read_pd_fth(FLAGS.input_path,FLAGS.filename_x)
        res = ffnn.predict(X)
        # Write Data
        df = pd.DataFrame.from_records(res)
        df.columns = ['c_'+str(i) for i in df.columns]
        to_pd_fth(FLAGS.output_path,FLAGS.model_name+'_results.fth',df)
        if FLAGS.input_path[:5]  == 's3://':
            ffnn.del_locals(FLAGS.model_name,FLAGS.stage)

    if FLAGS.stage == 'evaluate':
        #Load Data
        X = read_pd_fth(FLAGS.input_path,FLAGS.filename_x)
        y = Ffnn.read_y(FLAGS.input_path,FLAGS.filename_y,FLAGS.columns_y)

        res = ffnn.evaluate(X,y,FLAGS.metrics)
        # Write Results
        to_json(FLAGS.output_path,FLAGS.model_name+'_metrics.json',res)
        if FLAGS.input_path[:5]  == 's3://':
            ffnn.del_locals(FLAGS.model_name,FLAGS.stage)

if __name__ == '__main__':
    app.run(main)

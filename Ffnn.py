# Copyright (c) 2022 RWTH Aachen - Werkzeugmaschinenlabor (WZL)
# Contact: Simon Cramer, s.cramer@wzl-mq.rwth-aachen.de

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from absl import logging
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from tensorflow import keras
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import json
from typing import List, Dict, Tuple, Any
from zipfile import ZipFile
from s3_smart_open.filehandler import *
import shutil


tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(1)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

class Ffnn(BaseEstimator, RegressorMixin):
    """Simple feed forward neural network based on tf.keras in the form of a scikit-learn BaseEstimator.
    Support for 1D inputs and 1D outputs.

    Note:
        Regularisation on layers not yet supported.
    """
    def __init__(self,
                 input_shape: Tuple[int],
                 layers: List[Dict],
                 loss: str = 'MeanAbsoluteError',
                 metrics: List[str] = ['MeanSquaredError'],
                 optimizer_config: Dict[str,Any] = {'class_name': 'sgd','config':{}},
                 callbacks: Dict[str,Dict] = {'TensorBoard': {'log_dir': './logs'}},
                 batch_size: int = 32,
                 epochs: int = 100,
                 verbose: int = 2,
                 validation_freq: int = 1,
                 validation_split: float = 0.1,
                 buffer_size: int = 100,
                 model_input_path: str = '',
                 model_name: str = ''
                 ):
        """Initialize the simple feed forward neural network.

        Example:
        obj = Ffnn( (3,),
                    [50,25],
                    ['',''],
                    'MeanAbsoluteError',
                    ['MeanSquaredError'],
                    {'class_name': 'sgd'},
                    {'TensorBoard': {'log_dir': './logs'}},
                    100,
                    200,
                    1,
                    1,
                    0.2,
                    100)

        Args:
            input_shape (tuple): Shape of the expected input, e.g. (3,) for a 1D input.
            layers: List[Dict], Define the layers, cmp. tf.keras.layers.Dense.get_config()
            loss (str, optional): Loss for the training of the NN. Defaults to 'MeanAbsoluteError'.
            metrics (List[str], optional): List of metrics to use while training/evaluating the NN. Defaults to ['MeanSquaredError'].
            optimizer_config (Dict, optional): Dictionary describing the optimizer. Defaults to {'class_name': 'sgd','config':{}}.
            callbacks (Dict[Dict], optional): Callbacks for training. Example: {'Tensorboard': {'log_dir': './logs'}}. Defaults to {}.
            batch_size (int, optional): Batch size for training and preDictions. Defaults to 32.
            epochs (int, optional): Epochs for training. Defaults to 100.
            verbose (int, optional): Set verbosity level. 0 is no output. Defaults to 2.
            validation_freq (int, optional): After how many training epochs validation is run. To disable set to 0. Defaults to 1.
            validation_split (float, optional): Percentage of dataset to use for validation. Defaults to 0.1.
            buffer_size (int, optional): Buffer Size for shuffling of dataset between epochs. Defaults to 100.
        """
        self.layers = layers

        assert len(input_shape) > 0
        self.input_shape = input_shape

        assert batch_size > 1
        self.batch_size = batch_size
        assert epochs > 0
        self.epochs = epochs
        assert verbose in [0,1,2]
        self.verbose = verbose
        assert validation_freq >= 0
        self.validation_freq = validation_freq
        assert validation_split < 1
        assert validation_split >=0
        self.validation_split = validation_split
        assert buffer_size > 1
        self.buffer_size = buffer_size

        self.is_fitted_ = False

        policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
        tf.keras.mixed_precision.experimental.set_policy(policy) 

        self.callbacks = callbacks
        self.optimizer_config = optimizer_config
        self.loss = loss
        self.metrics = metrics
        self.model_input_path = model_input_path
        self.model_name = model_name
        # Check if values are correct without setting them
        self._initKerasFunctions()
    
    def _layerFromConfig(self, config: Dict):
        """Convert config in Json Format to Keras Layers.

        Args:
            config (Dict): Dictionary of a keras layer config. Dict has to contain 'type' key to indicate layer class to create.

         Returns:
             keras.layers Object of a keras.layers class
        """
        assert (type(config) is dict)

        class_name = config['type']
        config.pop('type', None)  # Remove type as it only indicates classnames

        layer_class = eval('tf.keras.layers.' + class_name)
        layer = layer_class.from_config(config)

        return layer

    def _initKerasFunctions(self):
        """Convert arguments passed as strings to tf.keras objects. Rely on tf.keras to check whether they exist.

        Returns:
            List[keras.layers]: List of keras layers
            List[keras.callbacks]: List of keras callback objects.
            keras.optimizer: Optimizer object
            keras.losses: Loss object
            List[keras.metrics]: List of metric objects
        """        
        layers = [self._layerFromConfig(config.copy()) for config in self.layers] # Ohne copy wird type von self.layers entfernt. Funktionsaufruf in __init__() und fit()

        callbacks = []
        if not len(self.callbacks) == 0:
            for name, config in self.callbacks.items():
                c = eval('keras.callbacks.'+name)()
                for param, value in config.items():
                    setattr(c,param,value)
                callbacks.append(c)
        # Terminate on NaN
        callbacks.append(keras.callbacks.TerminateOnNaN())

        optimizer = keras.optimizers.get(self.optimizer_config)
        loss = keras.losses.get(self.loss)
        metrics = [keras.metrics.get(m) for m in self.metrics]

        return layers, callbacks, optimizer, loss, metrics

    def _buildAndCompile(self,
                         layers: List,
                         optimizer: keras.optimizers,
                         loss: keras.losses,
                         metrics: List):
        """Build the tf.keras sequential model and compile it.

        Args:
            layers (List): list of tf.keras.layers
            optimizer (keras.optimizer): Optimizer object
            loss (keras.losses): Loss object
            metrics (List[keras.metrics]): List of metric objects
        """                           
        self.model = keras.Sequential()
        self.model.add(keras.Input(shape=self.input_shape))
        for layer in layers:
            self.model.add(layer)

        self.model.compile(optimizer = optimizer, loss=loss, metrics = metrics)

    def _makeTFData(self, X, y=None, validation_split: float=None, shuffle: bool=True):
        """Transform input data into tf.data.Dataset. Which is batched/prefechted.

        Args:
            X (ndarray, pd.Dataframe): Features
            y (ndarray, pd.Dataframe, optional): Targets. Defaults to None.
            validation_split (float, optional): Percentage of dataset to use for validation [0,1). Defaults to None.

        Returns:
            tf.data.dataset: Dataset for training/preDiction/evaluation
            tf.data.dataset: Dataset for validation
        """
        logging.info('Creating tf.dataset from '+str(X.shape)+' samples') 
        if type(X) == pd.DataFrame:
            X = X.values#.astype(np.float16)
               
        assert type(X) is np.ndarray, "Feature dataset passed in wrong format"      

        if not type(y) == type(None):
            if type(y) == pd.DataFrame:
                y = y.values#.astype(np.float16)
            assert type(y) is np.ndarray, "Target dataset passed in wrong format"

        val_dataset = None
        if not validation_split is None:
            validation_sample_count = int(np.ceil(X.shape[0] * validation_split))
            validation_index = np.random.choice(X.shape[0], validation_sample_count)
            
            x_val = X[validation_index,:]
            X = np.delete(X, validation_index, axis=0)

            val_tuple = (x_val,)

            if not type(y) == type(None):
                y_val = y[validation_index,:]         
                y = np.delete(y, validation_index, axis=0)
                val_tuple = val_tuple + (y_val,)

            val_dataset = tf.data.Dataset.from_tensor_slices(val_tuple)
            val_dataset = val_dataset.shuffle(buffer_size=self.buffer_size).batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        dataset_tuple = (X,)
        if not type(y) == type(None):
            dataset_tuple = dataset_tuple + (y,)
        dataset = tf.data.Dataset.from_tensor_slices(dataset_tuple)
        if shuffle == True:
            dataset = dataset.shuffle(buffer_size=self.buffer_size)
            logging.info('Shuffling dataset') 
        dataset = dataset.batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        return dataset, val_dataset

    def fit(self, X, y):
        """Either construct, compile and fit the model to a given dataset X (features) and y (targets), or warm-start a model with the possibility to adjust the optimizer and activation function.

        Args:
            X (array-like, sparse matrix): shape (n_samples, n_features). The training input samples.
            y (array-like): shape (n_samples,) or (n_samples, n_outputs). The target values (class labels in classification, real numbers in regression).

        Returns:
            Ffnn: Returns self.
        """
        self.layers.append({'type': 'Dense', 'units': y.shape[-1], 'use_bias': False, 'activation': 'linear'})
        layers, callbacks, optimizer, loss, metrics = self._initKerasFunctions()

        X, y = check_X_y(X, y, accept_sparse=False, multi_output=True)

        dataset, dataset_val = self._makeTFData(X,y,validation_split=self.validation_split, shuffle=True)

        if self.model_input_path != '':
            loaded_model = Ffnn.load(self.model_input_path, self.model_name)
            self.model = loaded_model.model
            for layer in range(len(self.model.layers)):
               self.model.layers[layer].activation = layers[layer].activation
            self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
            logging.info('Continue fitting with ' + str(X.shape) + ' samples')
        else:
            self._buildAndCompile(layers, optimizer, loss, metrics)
            logging.info('Begin fitting with ' + str(X.shape) + ' samples')

        self.model.fit(x=dataset, epochs=self.epochs, verbose=self.verbose, callbacks=callbacks, validation_data=dataset_val, shuffle=True, validation_freq=self.validation_freq)
        self.is_fitted_ = True

        return self

    def predict(self, X):
        """PreDict the targets given a set of inputs.

        Args:
            X (ndarray, pd.Dataframe): shape (n_samples, n_features). The input samples.

        Returns:
            numpy.ndarray: shape (n_samples, n_outputs). PreDicted values.
        """        
        X = check_array(X, accept_sparse=False)
        check_is_fitted(self, 'is_fitted_')
        
        dataset, _ = self._makeTFData(X,shuffle=False)

        return self.model.predict(X)

    def evaluate(self, X, y, metrics: List[str]):
        """Evaluate 

        Args:
            X (ndarray, pd.Dataframe): Features
            y (ndarray, pd.Dataframe): Targets
            metrics (List[str]): List of metrics to use while training/evaluating the NN

        Returns:
            [type]: [description]
        """
        if type(y) == pd.DataFrame:
            y=y.values        
        y_hat = self.predict(X)

        metrics = [keras.metrics.get(m) for m in metrics]

        res = {}
        for m in metrics:
            m.update_state(y, y_hat)
            res[m.name] = m.result().numpy().tolist()
        
        return res


    def save(self, path:str, name:str):
        """Saves the entire object to disk. As tf.keras.Model cannot be parsed by pickel, the model is saved first and excluded from the object.

        Args:
            path (str): Path on disk
            name (str): name of the model
        """

  

        tf.keras.models.save_model(
                self.model,
                "./model/001/",
                overwrite=True,
                signatures=None,
                include_optimizer=False
        )

        if not path[:5] == 's3://':
            self.model.save(os.path.join(path,name+'_weights'))  
        else:
            local_directory_to_s3(os.getcwd(), path, "model")
            self.model.save(os.path.join(os.getcwd(),name+'_weights'))
            with ZipFile(name+'_weights.zip','w') as outfile:
                for folderName, subfolders, filenames in os.walk(os.path.join(os.getcwd(),name+'_weights')):
                    for filename in filenames:
                        filepath = os.path.join(folderName,filename)
                        outfile.write(filepath,filepath.replace(os.getcwd(),''))
            to_s3(path,name+'_weights.zip',os.path.join(os.getcwd(),name+'_weights.zip'))

            if 'TensorBoard' in self.callbacks.keys():
                with ZipFile(name+'_logdir.zip', 'w') as zipObj2:
                    for folderName, subfolders, filenames in os.walk(os.path.join(os.getcwd(),self.callbacks['TensorBoard']['log_dir'])):
                        for filename in filenames:
                            filepath = os.path.join(folderName,filename)
                            zipObj2.write(filepath,filepath.replace(os.getcwd(),''))  
                to_s3(path,name+'_logdir.zip',os.path.join(os.getcwd(),name+'_logdir.zip'))

        self.model = name+'_weights'
        to_pckl(path,name+'_obj.pckl',self)


    @staticmethod
    def load(path:str, name:str):
        """Static class method to load model from disk. The object is unpickeld and the model is reloaded using tf.keras functionality.

        Args:
            path (str): Path on disk
            name (str): Name of the model

        Returns:
            Ffnn: Returns loaded Ffnn object (self).
        """  

        obj = read_pckl(path,name+'_obj.pckl')
        if not path[:5] == 's3://':
            model = keras.models.load_model(os.path.join(path,name+'_weights'))
        else:
            from_s3(path,name+'_weights.zip',os.getcwd())
            with ZipFile(os.path.join(os.getcwd(),name+'_weights.zip'),'r') as infile:
                infile.extractall(os.getcwd())
            model = keras.models.load_model(os.path.join(os.getcwd(),name+'_weights'))

        obj.model = model
        
        return obj
    
    def read_y(path:str , name:str, columns_y=None):
        y = read_pd_fth(path,name)
        logging.info('Y Head: {}'.format(y.columns.to_list()))
        if not columns_y is None:
            columns_to_drop = y.columns.to_list()
            for col in columns_y:
                if not col in y.columns:
                    logging.warning('Column {} not in dataframe {}. Cannot be kept.'.format(col, 'train_y.fth'))
                    continue
                columns_to_drop.remove(col)
            y.drop(columns=columns_to_drop,inplace=True)
            logging.info('Y Head: {}'.format(y.columns.to_list()))
        return y
    
    def del_locals(self, name, stage):
        if os.path.exists(os.path.join(os.getcwd(),name+'_weights.zip')):
            os.remove(name+'_weights.zip')
        else:
            logging.warning('{} does not exists!'.format(os.path.join(os.getcwd(),name+'_weights.zip')))

        if os.path.exists(os.path.join(os.getcwd(),name+'_logdir.zip')) and stage == 'fit':
            os.remove(name+'_logdir.zip')
        else:
            logging.warning('{} does not exists!'.format(os.path.join(os.getcwd(),name+'_logdir.zip')))

        if os.path.exists(os.path.join(os.getcwd(),name+'_weights')):
            shutil.rmtree(name+'_weights')
        else:
            logging.warning('{} does not exists!'.format(os.path.join(os.getcwd(),name+'_weights')))

        if os.path.exists(os.path.join(self.callbacks['TensorBoard']['log_dir'])) and stage == 'fit':
            shutil.rmtree(self.callbacks['TensorBoard']['log_dir'])
        else:
            logging.warning('{} does not exists!'.format(os.path.join(self.callbacks['TensorBoard']['log_dir'])))

        if os.path.exists(os.path.join(os.getcwd(), 'model')) and stage == 'fit':
            shutil.rmtree(os.path.join(os.getcwd(), 'model'))
        else:
            logging.warning('{} does not exists!'.format(os.path.join(os.getcwd(), 'model')))  

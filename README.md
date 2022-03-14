# Tensorflow FeedForward Neural Network
Simple feed forward neural network based on tf.keras in the form of a scikit-learn BaseEstimator. \
    Support for 1D inputs and 1D outputs.


## Installation

Clone the repository and install all requirements using `pip install -r requirements.txt` .


## Usage

You can run the code in two ways.
1. Use command line flags as arguments `python main.py --input_path= --output_path=...`
2. Use a flagfile.txt which includes the arguments `python main.py --flagfile=example/flagfile.txt`

### Input Flags/Arguments

#### --input_path
Specify the a local or s3 object storage path where the dataframe is stored.
For a s3 object storage path a valid s3 configuration yaml file is required.

#### --output_path
Specify the path where the profile report will be stored.
For a s3 object storage path a valid s3 configuration yaml file is required.

#### --model_input_path
Specify the path where models are loaded from to enable warm-starting models. 
For a s3 object storage path a valid s3 configuration yaml file is required.

#### --model_name
Name of the model to save/load.

#### --stage
Select the stage to execute on of the following options: fit, predict or evaluate 

#### --layers
Define the layers, cmp. tf.keras.layers.Dense.get_config(). 
Supply them as a list of dictionaries where to use double quotes ("") for each list element and single quotes ('') for dictionary entries. Otherwise the the list gets split after each comma. 
E.g. 
--layers="{'type': 'Dense', 'units': 15 , 'use_bias': False, 'activation': 'linear'}","{'type': 'Dense', 'units': 5 , 'use_bias': False, 'activation': 'linear'}","{'type': 'Dense', 'units': 3 , 'use_bias': False, 'activation': 'sigmoid'}"

#### --loss
Loss function for NN. See here for candidate functions: https://www.tensorflow.org/api_docs/python/tf/keras/losses

#### --metrics
Metrics for on training evaluation. See here for candidate functions: https://www.tensorflow.org/api_docs/python/tf/keras/metrics


#### --optimizer_config
Dict providing info about optimizer

#### --callbacks
Callbacks for training. Example: {"TensorBoard": {"log_dir": "./logs"}}. Path is relative to output_path.  See here for candidates: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks

#### --batch_size
Batch size for fit and predict

#### --epochs
Epochs for fit

#### --verbose
Choice of 0,1,2. See  https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit

#### --validation_freq
How often validation is done. https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit

#### --validation_split
Percentage of dataset for validation purposes. https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit

#### --buffer_size
Size of buffer for shuffling.


#### --columns_y
Name of the y column to train,predict

#### --set_gpus
Limit TF to only certain gpu.

#### --filename_x
Filename of _x.fth file

#### --filename_y
Filename of _y.fth file


## Example

First move to the repository directory.
Now you can run an example by using `python main.py --flagfile=example/flagfile_fit.txt`

## Data Set

The data set was recorded with the help of the Festo Polymer GmbH. The features (`x.csv`) are either parameters explicitly set on the injection molding machine or recorded sensor values. The target value (`y.csv`) is a crucial length measured on the parts. We measured with a high precision coordinate-measuring machine at the Laboratory for Machine Tools (WZL) at RWTH Aachen University.

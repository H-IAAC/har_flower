import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tfltransfer import bases
from tfltransfer import heads
from tfltransfer import optimizers
from tfltransfer.tflite_transfer_converter import TFLiteTransferConverter

import os

"""Define the base model.
To be compatible with TFLite Model Personalization, we need to define a
base model and a head model. 
Here we are using an identity layer for base model, which just passes the 
input as it is to the head model.
"""
def base_model(models_folder, input_dim):
    #TODO: checar esse input
    base = tf.keras.Sequential(
        [tf.keras.Input(shape=(None, input_dim)), tf.keras.layers.Lambda(lambda x: x)]
    )

    base.compile(loss="binary_crossentropy", optimizer="sgd")
    base.save(f"{models_folder}/identity_model", save_format="tf")
    return f"{models_folder}/identity_model"

def head_model(model_path):
    return tf.keras.models.load_model(model_path)



"""Convert the model for TFLite.
Using 10 classes in CIFAR10, learning rate = 1e-3 and batch size = 32
This will generate a directory called tflite_model with five tflite models.
Copy them in your Android code under the assets/model directory.
"""
def convert_to_tflite(models_path, tflite_name, head : tf.keras.Sequential, n_classes, train_batch_size, lr):
    base_path = bases.saved_model_base.SavedModelBase(os.path.join(models_path, 'identity_model'))
    #TODO: rever parametros
    converter = TFLiteTransferConverter(
        n_classes, base_path, heads.KerasModelHead(head), optimizers.SGD(lr), train_batch_size=train_batch_size
    )

    converter._enable_tflite_resource_variables = True
    '''Attention here!! The built-in flower converter is based on tf v1 and it has several flaws that were corrected in tf2.'''
    converter.convert_and_save(f"{models_path}/{tflite_name}")



def load_convert(models_folder, head_name):
    base_model_path = base_model(models_folder)
    head = head_model(os.path.join(models_folder, head_name))
    convert_to_tflite(models_folder, head)


def get_model_infos(model : tf.keras.Sequential):
    config = model.get_config() # Returns pretty much every information about your model
    print(config["layers"][0]["config"]["batch_input_shape"]) # returns a tuple of width, height and channels
    input_dim = config["layers"][0]["config"]["batch_input_shape"][1]
    n_classes = config["layers"][-1]['config']['units']

    return input_dim, n_classes


if __name__ == '__main__':
    models_folder = '/home/wander/OtherProjects/har_flower/notebooks/model'
    model_name = 'foldfold_0'
    train_batch_size = 10
    lr = 0.01

    head = head_model(f'{models_folder}/saved_model_{model_name}')
    input_dim, n_classes = get_model_infos(head)
    base_model_path = base_model(models_folder, input_dim)
    tflite_name  =  f'tflite_{model_name}'
    convert_to_tflite(models_folder, tflite_name, head, n_classes, train_batch_size, lr)
    pass
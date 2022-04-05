#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import tensorflow as tf
import es_utils as utils

labels = [
    'label:OR_standing',
    'label:SITTING',
    'label:LYING_DOWN',
    'label:FIX_running',
    'label:FIX_walking',
    'label:BICYCLING'
]

config = {
    'df_path': "../input/extrasensory/primary/0A986513-7828-4D53-AA1F-E02D6DF9561B.features_labels.csv",
    'neurons_1_base': 32,
    'neurons_1_head': 8,
    'neurons_2': 16,
    'labels': labels
}

har = utils.HAR(config)
har.run()


from tflite_convertor.tfltransfer import bases
from tflite_convertor.tfltransfer import heads
from tflite_convertor.tfltransfer import optimizers
from tflite_convertor.tfltransfer.tflite_transfer_converter import TFLiteTransferConverter

base_dir = "tflite_convertor/identity_model/saved_model_fold_"
target_dir = "tflite_convertor/tflite_model/saved_model_fold_"

for i in range(0, 5):
    base_path = bases.saved_model_base.SavedModelBase(base_dir+str(i))
    converter = TFLiteTransferConverter(
        # num_classes, base_model, head_model, optimizer, train_batch_size
        6, base_path, heads.KerasModelHead(har.head_model), optimizers.SGD(1e-1), train_batch_size=32
    )

    converter.convert_and_save(target_dir+str(i))

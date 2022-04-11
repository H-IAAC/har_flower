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
    'gen_base_model': False,
    'gen_head_model': True,
    'neurons_1_base': 32,
    'neurons_1_head': 8,
    'neurons_2': 16,
    'labels': labels
}

har = utils.HAR(config)

from tflite_convertor.tfltransfer import bases
from tflite_convertor.tfltransfer import heads
from tflite_convertor.tfltransfer import optimizers
from tflite_convertor.tfltransfer.tflite_transfer_converter import TFLiteTransferConverter

base_dir = "./model/saved_model_fold_"
target_dir = "../android/client/app/src/main/assets/model/fold_"

# Batch size usado na geracao do modelo base
batch_size = 50

for i in range(0, 5):
    base_model = bases.saved_model_base.SavedModelBase(base_dir+str(i))
    converter = TFLiteTransferConverter(
        # num_classes, base_model, head_model, optimizer, train_batch_size
        6, base_model, heads.KerasModelHead(har.head_model), optimizers.SGD(1e-1), train_batch_size=batch_size
    )

    converter.convert_and_save(target_dir+str(i))

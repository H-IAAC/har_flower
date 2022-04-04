#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
import es_utils as utils


# In[2]:


'''labels = ['label:LYING_DOWN',
 'label:SITTING',
 'label:FIX_walking',
 'label:FIX_running',
 'label:BICYCLING',
 'label:SLEEPING',
 'label:LAB_WORK',
 'label:IN_CLASS',
 'label:IN_A_MEETING',
 'label:LOC_main_workplace',
 'label:OR_indoors',
 'label:OR_outside',
 'label:IN_A_CAR',
 'label:ON_A_BUS',
 'label:DRIVE_-_I_M_THE_DRIVER',
 'label:DRIVE_-_I_M_A_PASSENGER',
 'label:LOC_home',
 'label:FIX_restaurant',
 'label:PHONE_IN_POCKET',
 'label:OR_exercise',
 'label:COOKING',
 'label:SHOPPING',
 'label:STROLLING',
 'label:DRINKING__ALCOHOL_',
 'label:BATHING_-_SHOWER',
 'label:CLEANING',
 'label:DOING_LAUNDRY',
 'label:WASHING_DISHES',
 'label:WATCHING_TV',
 'label:SURFING_THE_INTERNET',
 'label:AT_A_PARTY',
 'label:AT_A_BAR',
 'label:LOC_beach',
 'label:SINGING',
 'label:TALKING',
 'label:COMPUTER_WORK',
 'label:EATING',
 'label:TOILET',
 'label:GROOMING',
 'label:DRESSING',
 'label:AT_THE_GYM',
 'label:STAIRS_-_GOING_UP',
 'label:STAIRS_-_GOING_DOWN',
 'label:ELEVATOR',
 'label:OR_standing',
 'label:AT_SCHOOL',
 'label:PHONE_IN_HAND',
 'label:PHONE_IN_BAG',
 'label:PHONE_ON_TABLE',
 'label:WITH_CO-WORKERS',
 'label:WITH_FRIENDS']

len(labels)'''

labels = [
    'label:OR_standing',
    'label:SITTING',
    'label:LYING_DOWN',
    'label:FIX_running',
    'label:FIX_walking',
    'label:BICYCLING'
]

"""
paths = ['../input/extrasensory/primary/0A986513-7828-4D53-AA1F-E02D6DF9561B.features_labels.csv',
         '../input/extrasensory/primary/0BFC35E2-4817-4865-BFA7-764742302A2D.features_labels.csv',
         '../input/extrasensory/primary/0BFC35E2-4817-4865-BFA7-764742302A2D.features_labels.csv',
         '../input/extrasensory/primary/0BFC35E2-4817-4865-BFA7-764742302A2D.features_labels.csv']
bas = []

config = {'df_path': '../input/user1.features_labels.csv'}

for path in paths:

    config = {
            'df_path': path,
            'neurons_1' : 32, 
            'neurons_2' : 16, 
            'labels': labels
    }

    har = utils.HAR(config)
    test_results, ba = har.run()
    #model, best_hps, best_epoch, test_results, ba = har.hypertunning()
    bas.append(ba.numpy())
    

print(bas)
print(max(bas))


paths = ['../input/extrasensory/primary/0A986513-7828-4D53-AA1F-E02D6DF9561B.features_labels.csv', 
         '../input/extrasensory/primary/0BFC35E2-4817-4865-BFA7-764742302A2D.features_labels.csv',
         '../input/extrasensory/primary/0BFC35E2-4817-4865-BFA7-764742302A2D.features_labels.csv',
         '../input/extrasensory/primary/0BFC35E2-4817-4865-BFA7-764742302A2D.features_labels.csv']
bas = []

for path in paths:

    config = {
            'df_path': path,
            #'df_path': '../full_data/exp_/fold_0/raw_40.csv',
            #'neurons_1' : 32, 
            #'neurons_2' : 16, 
            'labels': labels
    }

    har = utils.HAR(config)
    #test_results, ba = har.run()
    model, best_hps, best_epoch, test_results, ba = har.hypertunning()
    bas.append(ba.numpy())
    

print(bas)



bas = []
config = {
        #'df_path': path,
        'df_path': '../full_data/exp_/fold_0/raw_40.csv',
        'neurons_1' : 32, 
        'neurons_2' : 16, 
        'labels': labels
}

har = utils.HAR(config)
test_results, ba = har.run()
#model, best_hps, best_epoch, test_results, ba = har.hypertunning()
bas.append(ba.numpy())


print(bas)
print(max(bas))


import glob
bas = []
folderpath = '/home/wander/OtherProjects/har_flower/full_data'
for path in glob.iglob(f'{folderpath}/**.csv', recursive=True):
    config = {
            'df_path': path,
            #'df_path': '../full_data/exp_/fold_0/raw_40.csv',
            #'neurons_1' : 32, 
            #'neurons_2' : 16, 
            'labels': labels
    }

    har = utils.HAR(config)
    #test_results, ba = har.run()
    model, best_hps, best_epoch, test_results, ba = har.hypertunning()
    bas.append(ba.numpy())
    

print(bas)
"""

config = {
    'df_path': "../input/extrasensory/primary/0A986513-7828-4D53-AA1F-E02D6DF9561B.features_labels.csv",
    'neurons_1_base': 32,
    'neurons_1_head': 8,
    'neurons_2': 16,
    'labels': labels
}

har = utils.HAR(config)
har.run()


# In[6]:

from tflite_convertor.tfltransfer import bases
from tflite_convertor.tfltransfer import heads
from tflite_convertor.tfltransfer import optimizers
from tflite_convertor.tfltransfer.tflite_transfer_converter import TFLiteTransferConverter


# In[21]:


base_model = har.base_model
base_model.save("tflite_convertor/identity_model", save_format="tf")


# In[23]:

base_path = bases.saved_model_base.SavedModelBase("tflite_convertor/identity_model")
converter = TFLiteTransferConverter(
    # num_classes, base_model, head_model, optimizer, train_batch_size
    6, base_path, heads.KerasModelHead(har.head_model), optimizers.SGD(1e-1), train_batch_size=32
)

converter.convert_and_save("tflite_convertor/tflite_model")


# In[ ]:





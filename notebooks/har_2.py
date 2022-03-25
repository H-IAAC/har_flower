#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np
import pandas as pd
import tensorflow as tf
import es_utils as utils


# In[20]:


config = {'df_path': '../input/user1.features_labels.csv'}
har = utils.HAR(config)
har.dummy_test()


# In[3]:


data = pd.read_csv('../input/user1.features_labels.csv')#.dropna() #TODO: attention

#data.isna().sum()
data.drop(columns=['lf_measurements:temperature_ambient'], inplace=True)
data[data.columns.drop(data.filter(regex='label:'))]


# In[4]:


#data[data.columns.drop(data.filter(regex='label:'))]
data.filter(regex='label:').columns.values.tolist()


# In[5]:


data[data.columns.drop(data.filter(regex='label:'))].dropna()


# In[6]:


data.fillna(0.0, inplace=True)
#data.info()
data[data.columns.drop(data.filter(regex='label:'))].dropna()


# ### Convert the model for TFLite.

# #### This will generate a directory called tflite_model with five tflite models.
# #### Copy them in your Android code under the assets/model directory.

# In[7]:


from tflite_convertor.tfltransfer import bases
from tflite_convertor.tfltransfer import heads
from tflite_convertor.tfltransfer import optimizers
from tflite_convertor.tfltransfer.tflite_transfer_converter import TFLiteTransferConverter


# In[21]:


model = har.base_model
model.save("tflite_convertor/identity_model", save_format="tf")


# In[23]:


base_path = bases.saved_model_base.SavedModelBase("tflite_convertor/identity_model")
converter = TFLiteTransferConverter(
    4, base_path, heads.KerasModelHead(har.head_model), optimizers.SGD(1e-1), train_batch_size=10
)

converter.convert_and_save("tflite_model")


# In[ ]:





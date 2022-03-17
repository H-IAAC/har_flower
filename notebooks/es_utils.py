from cProfile import label
import numpy  as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#from tensorflow.keras.utils import to_categorical
import keras_tuner as kt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
#from skmultilearn.model_selection import iterative_train_test_split

import glob
import json
import random
from functools import partial
import os

class MLPMultilabel:
    def __init__(self, input_dim, num_classes, neurons_1=16, neurons_2=None, l2=0.01) -> None:
        self.model  = self.build_model(input_dim, num_classes, neurons_1=neurons_1, neurons_2=neurons_2, l2=l2)

    def train(self, x_train_norm, y_train):
        self.model.fit(x_train_norm, y_train, epochs=40, batch_size=10, verbose=1, validation_split=0.2)

    def evaluate(self, x_test_norm, y_test):
        test_results = self.model.evaluate(x_test_norm, y_test, verbose=1)
        ba = self.test_BA(x_test_norm, y_test)
        print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]}%')
        print(f'Averaged Balanced Accuracy: {ba:.6f}')
        
        return test_results, ba

    def predict(self, x):
        return self.model.predict(x)

    def test_BA(self, x_test_norm, y_test):
        y_pred = self.model.predict(x_test_norm)
        return avg_multilabel_BA(y_test, y_pred)

    def build_model(self, input_dim, num_classes, neurons_1=16, neurons_2=None, l2=0.01) -> Sequential:
        model = Sequential()
        model.add(Dense(neurons_1, input_dim=input_dim, activation='relu', activity_regularizer=tf.keras.regularizers.l2(l2)))
        if neurons_2 is not None:
            model.add(Dense(neurons_2, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))

        # Configure the model and start training
        sgd = tf.keras.optimizers.SGD(learning_rate=0.1, decay=1e-2, momentum=0.5)
        model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['categorical_accuracy'])
        return model

    
    def hypertunning(self, x_train_norm, y_train, n_hidden=1):
        #if n_hidden == 2:
        #    partial_hyper = partial(self.hyper_model_2_build, input_dim=x_train_norm.shape[1], num_classes=y_train.shape[1])
        #else:
        partial_hyper = partial(self.hyper_model_build, input_dim=x_train_norm.shape[1], num_classes=y_train.shape[1])
            #partial_hyper = partial(self.hyper_model_1_build, input_dim=x_train_norm.shape[1], num_classes=y_train.shape[1])
        tuner = kt.Hyperband(partial_hyper,
                     objective='val_categorical_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='hypertunning',
                     project_name='experimento_1')
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

        tuner.search(x_train_norm, y_train, epochs=50, validation_split=0.2, callbacks=[stop_early])
        # Get the optimal hyperparameters
        best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

        ##print(f"""
        ##The hyperparameter search is complete. The optimal number of units in the first densely-connected
        ##layer is {best_hps.get('units_1')} and the optimal learning rate for the optimizer
        ##is {best_hps.get('learning_rate')}.
        ##""")

        # Build the model with the optimal hyperparameters and train it on the data for 50 epochs
        model = tuner.hypermodel.build(best_hps)
        history = model.fit(x_train_norm, y_train, epochs=50, validation_split=0.2)

        val_acc_per_epoch = history.history['val_categorical_accuracy']
        best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
        print('Best epoch: %d' % (best_epoch,))

        hypermodel = tuner.hypermodel.build(best_hps)

        # Retrain the model
        hypermodel.fit(x_train_norm, y_train, epochs=best_epoch, batch_size=10, validation_split=0.2)

        self.model = hypermodel

        return hypermodel, best_hps, best_epoch

    
    def hyper_model_build(self, hp, input_dim, num_classes) -> Sequential:
        model = Sequential()

        hp_units = hp.Int('units_1', min_value=4, max_value=64, step=4)
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3, 1e-4])
        hp_momentum = hp.Choice('momentum', values=[0.8, 0.5, 0.3, 0.1])
        hp_l2 = hp.Choice('l2', values=[1e-2, 1e-3, 1e-4])

        model.add(Dense(units=hp_units, input_dim=input_dim, activation='relu', activity_regularizer=tf.keras.regularizers.l2(hp_l2)))

        for i in range(1, hp.Int("num_layers", 1, 2)):
            model.add(
                Dense(
                    units=hp.Int(f"units_{i+1}", min_value=4, max_value=64, step=4),
                    activation='relu', activity_regularizer=tf.keras.regularizers.l2(hp_l2))
                )


        model.add(Dense(num_classes, activation='softmax'))

        # Configure the model and start training
        sgd = tf.keras.optimizers.SGD(learning_rate=hp_learning_rate, decay=1e-2, momentum=hp_momentum)
        model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['categorical_accuracy'])
        return model


    '''def hyper_model_1_build(self, hp, input_dim, num_classes) -> Sequential:
        model = Sequential()

        hp_units = hp.Int('units', min_value=4, max_value=64, step=4)
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3, 1e-4])
        hp_momentum = hp.Choice('momentum', values=[0.8, 0.5, 0.3, 0.1])
        hp_l2 = hp.Choice('l2', values=[1e-2, 1e-3, 1e-4])

        model.add(Dense(units=hp_units, input_dim=input_dim, activation='relu', activity_regularizer=tf.keras.regularizers.l2(hp_l2)))
        model.add(Dense(num_classes, activation='softmax'))

        # Configure the model and start training
        sgd = tf.keras.optimizers.SGD(learning_rate=hp_learning_rate, decay=1e-2, momentum=hp_momentum)
        model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['categorical_accuracy'])
        return model
    
    def hyper_model_2_build(self, hp, input_dim, num_classes) -> Sequential:
        model = Sequential()

        hp_units_1 = hp.Int('units_1', min_value=4, max_value=64, step=4)
        hp_units_2 = hp.Int('units_2', min_value=4, max_value=64, step=4)
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3, 1e-4])
        hp_momentum = hp.Choice('momentum', values=[0.8, 0.5, 0.3, 0.1])
        hp_l2 = hp.Choice('l2', values=[1e-2, 1e-3, 1e-4])

        model.add(Dense(units=hp_units_1, input_dim=input_dim, activation='relu', activity_regularizer=tf.keras.regularizers.l2(hp_l2)))
        model.add(Dense(hp_units_2, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))

        # Configure the model and start training
        sgd = tf.keras.optimizers.SGD(learning_rate=hp_learning_rate, decay=1e-2, momentum=hp_momentum)
        model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['categorical_accuracy'])
        return model'''

    
        

class DataProcessingExtrasensory:
    def __init__(self, raw: pd.DataFrame, x=None, y=None, labels=None, dropna=True) -> None:
        raw = raw.dropna(subset=labels)
        self.x, self.y = self.get_x_y_from_raw(raw)
        if labels is not None:
            self.y = self.select_labels(self.y, labels)
        self.x_train, self.x_test, self.y_train, self.y_test = self.split_train_test()
    
    def split_train_test(self, test_size=0.25):
        #x_train, x_test, y_train, y_test = iterative_train_test_split(self.x, self.y, test_size=test_size)
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=test_size, random_state=42)

        min_max_scaler = preprocessing.MinMaxScaler()
        input_shape = x_train.shape
        
        x_train.to_numpy().reshape(input_shape[0], input_shape[1])
        x_train = pd.DataFrame(min_max_scaler.fit_transform(x_train))
        
        #num_classes = y_train.shape[1]
        #y_train = to_categorical(y_train, num_classes=num_classes)

        x_test = pd.DataFrame(min_max_scaler.transform(x_test))
        #y_test = to_categorical(y_test,num_classes = num_classes)

        return x_train, x_test, y_train, y_test


    def get_x_y_from_raw(self, raw):
        raw = self.treat_missing(raw) #TODO: attention
        x = raw[raw.columns.drop(raw.filter(regex='label:'))]
        y = raw.filter(regex='label:')
        return x, y
    
    def select_labels(self, y, labels : list):
        return y[labels]

    def treat_missing(self, data: pd.DataFrame):
        #default
        return data.fillna(0.0)


class HAR:
    def __init__(self, config : dict) -> None:
        self.config = {
            'df_path': None,
            'df': None,
            'hypertunning': False,
            'hypertunning_params': {},
            'neurons_1' : 16, 
            'neurons_2' : None, 
            'l2' : 0.01,
            'labels' : ['label:SITTING', 'label:LYING_DOWN','label:OR_standing', 'label:FIX_walking']
        }
        
        for key in config:
            self.config[key] = config[key]
        # TODO: checar se todos os parâmetros tão aqui ou inicializar
        
        if self.config['df'] is None:
            self.data= DataProcessingExtrasensory(self.load(self.config['df_path']), labels=self.config['labels']) #TODO: attention
        else:
            self.data= DataProcessingExtrasensory(self.config['df'], labels=self.config['labels']) #TODO: attention
        
        if not self.config['hypertunning']:
            self.mlp = self.make_mlp(
                self.data.x_train.shape[1], 
                self.data.y_train.shape[1], 
                neurons_1=self.config['neurons_1'], 
                neurons_2=self.config['neurons_2'], 
                l2=self.config['l2'])
        
        else:
            self.hypertunning()



    def make_mlp(self, input_dim, num_classes, neurons_1, neurons_2, l2):
        return MLPMultilabel(input_dim, num_classes, neurons_1=neurons_1, neurons_2=neurons_2, l2=l2)

    def load_data(self):
        pass

    def load(self, df_path):
        #return pd.read_csv('../input/user1.features_labels.csv')#.dropna()
        return pd.read_csv(df_path)

    def run(self):
        self.mlp.train(self.data.x_train, self.data.y_train)
        self.mlp.evaluate(self.data.x_test, self.data.y_test)

    #TODO: test
    def hypertunning(self):
        results = {}
        '''model, best_hps, best_epoch = self.mlp.hypertunning(self.data.x_train, self.data.y_train)
        test_results, ba = self.mlp.evaluate(self.data.x_test, self.data.y_test)
        results['1_hidden'] = [model, best_hps, best_epoch, test_results, ba]

        model, best_hps, best_epoch = self.mlp.hypertunning(self.data.x_train, self.data.y_train, n_hidden=2)
        test_results, ba = self.mlp.evaluate(self.data.x_test, self.data.y_test)
        results['2_hidden'] = [model, best_hps, best_epoch, test_results, ba]

        if results['1_hidden'][4] > results['2_hidden'][4]: #1 hidden layer with better BA
            model, best_hps, best_epoch, test_results, ba = results['1_hidden']
            self.mlp.model = model
        '''

        model, best_hps, best_epoch = self.mlp.hypertunning(self.data.x_train, self.data.y_train)
        test_results, ba = self.mlp.evaluate(self.data.x_test, self.data.y_test)
        #results['1_hidden'] = [model, best_hps, best_epoch, test_results, ba]


        return model, best_hps, best_epoch, test_results, ba
        
    def evaluate(self):
        self.mlp.evaluate(self.data.x_test, self.data.y_test)


def avg_multilabel_BA(y_true, y_pred):
    ba_array = []
    print(y_pred.shape[1])
    for i in range(y_pred.shape[1]):
        report = classification_report(y_true.to_numpy()[:, i], (y_pred[:, i] > 0.5), output_dict=True, zero_division=0)
        sensitivity = report['1.0']['recall'] # tp / (tp + fn)
        specificity = report['0.0']['recall'] #specificity = tn / (tn+fp)
        ba = 0.5*(specificity+sensitivity)
        ba_array.append(ba)
    return np.mean(ba_array)


def testar():
    config = {'df_path': '/home/wander/OtherProjects/har_flower/input/user1.features_labels.csv'}
    har = HAR(config)
    har.dummy_test()


def get_all_user_csvs(folderpath : str):
    answer = []
    for filename in glob.iglob(f'{folderpath}/**.csv', recursive=True):
        answer.append(filename)
    return answer


def create_k_folds_n_users(k_folds: int, n_users: int, folderpath: str):
    all_csvs = get_all_user_csvs(folderpath)
    all_dfs = {}
    for i in range(k_folds):
        fold_list_train = random.sample(all_csvs, n_users)
        fold_df_train = pd.read_csv(fold_list_train[0])
        for csv in fold_list_train[1:]:
            fold_df_train = fold_df_train.append(pd.read_csv(csv))

        fold_list_test = np.setdiff1d(all_csvs, fold_list_train)
        #fold_df_test = pd.read_csv(fold_list_test[0])
        #for csv in fold_list_test[1:]:
        fold_list_test = [csv.split(f'{folderpath}/')[1] for csv in fold_list_test]

            #fold_df_test = fold_df_test.append(pd.read_csv(csv))
        #print(fold_list_test)
        
        # Path
        path_exp = os.path.join(folderpath, f'exp_/fold_{i}')
        try:
            os.mkdir(path_exp)
        except FileExistsError as e:
            pass

        for test_user in fold_list_test:
            user_id = test_user.split('.features_labels.csv')[0]

            path_user = os.path.join(path_exp, f'{user_id}')
            try:
                os.mkdir(path_user)
            except FileExistsError as e:
                pass

            raw = pd.read_csv(os.path.join(folderpath, test_user)).fillna(0.0)
            x = raw[raw.columns.drop(raw.filter(regex='label:'))]
            y = raw.filter(regex='label:')
            x_train, x_test, y_train, y_test  = train_test_split(x, y, test_size=0.2, random_state=42)
            x_train.to_csv(f'{path_user}/x_train.csv')
            x_test.to_csv(f'{path_user}/x_test.csv')
            y_train.to_csv(f'{path_user}/y_train.csv')
            y_test.to_csv(f'{path_user}/y_test.csv')
    
        #with open(f'fold_{i}_test_cvs.json', 'w+') as json_data:  # abrir o fields
        #    json.dump(fold_list_test, json_data)
        #    results.append(json.load(json_data)) 
        #pessima ideia --> muito espaço em disco
        #fold_df_train.to_csv(f'fold_{i}_train.csv')
        #fold_df_test.to_csv(f'fold_{i}_test.csv')
        all_dfs[f'fold_{i}'] = {'train': fold_df_train}#, 'test': fold_df_test}

    return all_dfs

if __name__ == '__main__':
    pass
    a = create_k_folds_n_users(5, 3, '/home/wander/OtherProjects/har_flower/sample_data')
    #a = create_k_folds_n_users(5, 40, '/home/wander/OtherProjects/har_flower/full_data')
    #print(a['fold_0'])
    #testar()
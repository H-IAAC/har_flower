
import os
import glob
import random
from functools import partial

import numpy  as np
import pandas as pd
from sklearn.utils import shuffle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.initializers import GlorotNormal
from tensorflow import math
from tensorflow import where, stack, zeros_like, boolean_mask
from tensorflow.keras.backend import any 
from tensorflow.keras.models import load_model
import keras_tuner as kt
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy
from tensorflow.keras.metrics import AUC, TruePositives, TrueNegatives, FalseNegatives, FalsePositives

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
#from skmultilearn.model_selection import iterative_train_test_split
from skmultilearn.model_selection import IterativeStratification
from sklearn.model_selection import KFold


tp = TruePositives()
tn = TrueNegatives()
fp = FalsePositives()
fn = FalseNegatives()

class MLPMultilabel:
    def __init__(self, input_dim, num_classes, neurons_1=32, neurons_2=None, l2_val=0.01) -> None:
        self.model  = self.build_model(input_dim, num_classes, neurons_1=neurons_1, neurons_2=neurons_2, l2_val=l2_val)

    def train(self, x_train_norm, y_train):
        self.model.fit(x_train_norm, y_train, epochs=40, batch_size=10, verbose=1, validation_split=0.2)

    def evaluate(self, x_test_norm, y_test):
        test_results = self.model.evaluate(x_test_norm, y_test, verbose=1)
        ba = self.test_BA(x_test_norm, y_test)
        print(f'Test results - Loss: {test_results[0]} - Averaged Balanced Accuracy: {test_results[1]}%')
        print(f'Averaged Balanced Accuracy: {ba:.6f}')
        
        return test_results, ba

    def predict(self, x):
        return self.model.predict(x)

    def test_BA(self, x_test_norm, y_test):
        y_pred = self.model.predict(x_test_norm)
        return avg_multilabel_BA_2(y_test, y_pred)

    def load_model(self, model_path):
        self.model = load_model(model_path, custom_objects={"avg_multilabel_BA_2": avg_multilabel_BA_2} )

    def build_model(self, input_dim, num_classes, neurons_1=32, neurons_2=None, l2_val=0.01) -> Sequential:
        model = Sequential()
        initializer = GlorotNormal()
        model.add(Dense(neurons_1, input_dim=input_dim, activation='relu', kernel_initializer=initializer))
        #model.add(Dense(neurons_1, input_dim=input_dim, activation='relu', activity_regularizer=l2(l2_val)))
        if neurons_2 is not None:
            model.add(Dense(neurons_2, activation='relu', kernel_initializer=initializer))
        #model.add(Dropout(.2))
        model.add(Dense(num_classes, activation='sigmoid', kernel_initializer=initializer))

        # Configure the model and start training
        sgd = SGD(learning_rate=0.1, decay=1e-2, momentum=0.5)
        adam = Adam(learning_rate=0.1)
        model.compile(loss='binary_crossentropy', optimizer=adam, metrics=[avg_multilabel_BA_2])#metrics=[AUC(from_logits=True)])
        #model.compile(loss=nan_bce, optimizer=adam, metrics=['categorical_accuracy'])
        return model

    
    def hypertunning(self, x_train_norm, y_train,):
        
        partial_hyper = partial(self.hyper_model_build, input_dim=x_train_norm.shape[1], num_classes=y_train.shape[1])
        #tuner = kt.Hyperband(partial_hyper,
        #             objective= kt.Objective('val_avg_multilabel_BA_2', direction="max"),#'val_avg_multilabel_BA_2',
        #             max_epochs=10,
        #             factor=3,
        #             directory='hypertunning',
        #             project_name='experimento_1')
        tuner = kt.RandomSearch(partial_hyper,
                     objective= kt.Objective('val_avg_multilabel_BA_2', direction="max"),#'val_avg_multilabel_BA_2',
                     max_trials=100,
                     executions_per_trial=1,
                     directory='hypertunning',
                     project_name='experimento_1_RandSearchCV')
        
        stop_early = EarlyStopping(monitor='val_loss', patience=5)

        tuner.search(x_train_norm, y_train, epochs=100, validation_split=0.2, callbacks=[stop_early])
        # Get the optimal hyperparameters
        best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

        model = tuner.hypermodel.build(best_hps)
        history = model.fit(x_train_norm, y_train, epochs=100, batch_size=50, shuffle=True, validation_split=0.2)

        val_acc_per_epoch = history.history['val_avg_multilabel_BA_2']
        best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
        print('Best epoch: %d' % (best_epoch,))

        hypermodel = tuner.hypermodel.build(best_hps)
        # Retrain the model
        hypermodel.fit(x_train_norm, y_train, epochs=best_epoch, batch_size=50, shuffle=True, validation_split=0.2)

        self.model = hypermodel
        return hypermodel, best_hps, best_epoch

    
    def hyper_model_build(self, hp, input_dim, num_classes) -> Sequential:
        model = Sequential()
        initializer = GlorotNormal()

        hp_units = hp.Int('units_1', min_value=4, max_value=64, step=4)
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3, 1e-4])
        #hp_momentum = hp.Choice('momentum', values=[0.8, 0.5, 0.3, 0.1])
        #hp_l2 = hp.Choice('l2', values=[0.0, 1e-2, 1e-3, 1e-4])

        model.add(Dense(units=hp_units, input_dim=input_dim, activation='relu', kernel_initializer=initializer))

        for i in range(1, hp.Int("num_layers", 1, 2)):
            model.add(
                Dense(
                    units=hp.Int(f"units_{i+1}", min_value=4, max_value=64, step=4),
                    activation='relu', kernel_initializer=initializer) #, activity_regularizer=l2(hp_l2)
                )


        model.add(Dense(num_classes, activation='sigmoid',kernel_initializer=initializer))

        # Configure the model and start training
        #sgd = SGD(learning_rate=hp_learning_rate, decay=1e-2, momentum=hp_momentum)
        adam = Adam(learning_rate=hp_learning_rate)
        #adam = Adam(learning_rate=hp_learning_rate, epsilon=hp_momentum)
        model.compile(loss='binary_crossentropy', optimizer=adam, metrics=[avg_multilabel_BA_2])#metrics=['categorical_accuracy'])
        #model.compile(loss=nan_bce, optimizer=adam, metrics=['categorical_accuracy'])
        return model


## -------------------------------        
class DataProcessingExtrasensory:
    def __init__(self, raw: pd.DataFrame, x=None, y=None, labels=None, dropna=True) -> None:
        #raw = raw.dropna(subset=labels)
        self.x, self.y = self.get_x_y_from_raw(raw)
        if labels is not None:
            self.y = self.select_labels(self.y, labels)
        #self.x_train, self.x_test, self.y_train, self.y_test = self.iterative_split_train_test() #self.split_train_test()
        self.x_train, self.x_test, self.y_train, self.y_test = self.split_train_test()
    
    def split_train_test(self, test_size=0.20):
        #x_train, x_test, y_train, y_test = iterative_train_test_split(self.x, self.y, test_size = test_size)
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=test_size, random_state=42)

        min_max_scaler = preprocessing.MinMaxScaler()
        input_shape = x_train.shape
        
        x_train.to_numpy().reshape(input_shape[0], input_shape[1])
        x_train = pd.DataFrame(min_max_scaler.fit_transform(x_train))
        

        x_test = pd.DataFrame(min_max_scaler.transform(x_test))

        return x_train, x_test, y_train, y_test

    def iterative_split_train_test(self, train_size=0.8):
        """Custom iterative train test split which
        'maintains balanced representation with respect
        to order-th label combinations.'
        """
        stratifier = IterativeStratification(
            n_splits=2, order=1, sample_distribution_per_fold=[1.0-train_size, train_size, ])
        train_indices, test_indices = next(stratifier.split(self.x, self.y))
        X_train, y_train = self.x.iloc[train_indices], self.y.iloc[train_indices]
        X_test, y_test = self.x.iloc[test_indices], self.y.iloc[test_indices]
        return X_train, X_test, y_train, y_test


    def get_x_y_from_raw(self, raw):
        #raw = self.treat_missing(raw) #TODO: attention
        #raw = raw.fillna(0.0)
        raw.drop(columns=['timestamp'], inplace=True)
        x = raw[raw.columns.drop(raw.filter(regex='label:'))]
        y = raw.filter(regex='label:')
        x = self.treat_missing(x)
        y = self.treat_missing(y)
        #y = y.fillna(0) #TODO: erase
        return x, y
    
    def select_labels(self, y, labels : list):
        return y[labels]

    def treat_missing(self, data: pd.DataFrame):
        #default
        ##imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        ##return imputer.fit_transform(data)
        #return data.fillna(data.median())
        return data.fillna(0.0)


## -------------------------------
class HAR:
    def __init__(self, config : dict) -> None:
        self.config = {
            'df_path': None,
            'df': None,
            'hypertunning': False,
            'from_saved': None,
            'hypertunning_params': {},
            'neurons_1' : 32, 
            'neurons_2' : None, 
            'l2' : 0.01,
            'labels' : ['label:SITTING', 'label:LYING_DOWN','label:OR_standing', 'label:FIX_walking']
        }
        
        for key in config:
            self.config[key] = config[key]
        
        if self.config['df'] is None:
            self.data= DataProcessingExtrasensory(self.load(self.config['df_path']), labels=self.config['labels']) #TODO: attention
        else:
            self.data= DataProcessingExtrasensory(self.config['df'], labels=self.config['labels']) #TODO: attention
        
        if not self.config['hypertunning'] and self.config['from_saved'] is None:
            self.mlp = self.make_mlp(
                self.data.x_train.shape[1], 
                self.data.y_train.shape[1], 
                neurons_1=self.config['neurons_1'], 
                neurons_2=self.config['neurons_2'], 
                l2_val=self.config['l2'])
        
        elif self.config['from_saved'] is not None:
            self.mlp = self.from_saved(
                self.config['from_saved'], 
                self.data.x_train.shape[1], 
                self.data.y_train.shape[1], 
                neurons_1=self.config['neurons_1'], 
                neurons_2=self.config['neurons_2'], 
                l2_val=self.config['l2'])
        else:
            self.hypertunning()


    def make_mlp(self, input_dim, num_classes, neurons_1, neurons_2, l2_val):
        return MLPMultilabel(input_dim, num_classes, neurons_1=neurons_1, neurons_2=neurons_2, l2_val=l2_val)

    def load_data(self):
        pass

    def load(self, df_path):
        return pd.read_csv(df_path)

    def run(self):
        self.mlp.train(self.data.x_train, self.data.y_train)
        test_results, ba = self.mlp.evaluate(self.data.x_test, self.data.y_test)
        return test_results, ba

    def hypertunning(self):
        model, best_hps, best_epoch = self.mlp.hypertunning(self.data.x_train, self.data.y_train)
        test_results, ba = self.mlp.evaluate(self.data.x_test, self.data.y_test)

        return model, best_hps, best_epoch, test_results, ba
    

    def from_saved(self, model_path, input_dim, num_classes, neurons_1, neurons_2, l2_val):
        mlp = MLPMultilabel(input_dim, num_classes, neurons_1=neurons_1, neurons_2=neurons_2, l2_val=l2_val)
        mlp.load_model(model_path)
        return mlp

    def evaluate(self):
        self.mlp.evaluate(self.data.x_test, self.data.y_test)



## ------------------------- Functions not in a class-----------------------------------------------
def avg_multilabel_BA(y_truei, y_predi):
    ba_array = []
    

    for i in range(y_predi.shape[1]):
        #y_true, y_pred = remove_nans_np(y_truei, y_predi)
        try:
            y_true, y_pred = remove_nans_np(y_truei.to_numpy()[:, i], y_predi[:, i])
            report = classification_report(y_true, (y_pred > 0.5), output_dict=True, zero_division=0)
        except:
            y_true, y_pred = remove_nans_np(y_truei[:, i], y_predi[:, i])
            report = classification_report(y_true, (y_pred > 0.5), output_dict=True, zero_division=0)
        #sensitivity = report['1.0']['recall'] # tp / (tp + fn)
        try:
            specificity = report['0.0']['recall'] #specificity = tn / (tn+fp)
        except:
            specificity = 1
        try:
            sensitivity = report['1.0']['recall'] # tp / (tp + fn)
        except:
            sensitivity = specificity # tp / (tp + fn)
        ba = 0.5*(specificity+sensitivity)
        ba_array.append(ba)
    return np.mean(ba_array)


def avg_multilabel_BA_2(y_truei, y_predi):
    ba_array = []
    
    global tp
    global tn
    global fp
    global fn
    tp.update_state(y_truei, y_predi)
    tn.update_state(y_truei, y_predi)
    fp.update_state(y_truei, y_predi)
    fn.update_state(y_truei, y_predi)

    specificity = math.divide(tn.result(), math.add(tn.result(), fp.result())) #tn / (tn+fp)
    sensitivity = math.divide(tp.result(), math.add(tp.result(), fn.result())) #tp / (tp + fn)
    ba = math.multiply(0.5, math.add(specificity, sensitivity))#0.5*(specificity+sensitivity)
    return ba



def remove_nans_np(x, y):
    mask = ~np.isnan(x) & ~np.isnan(y)
    #mask = math.logical_and(math.logical_not(math.is_nan(x)), math.logical_not(math.is_nan(y)))
    
    marra = np.ma.MaskedArray(x, mask=~mask)
    marrb = np.ma.MaskedArray(y, mask=~mask)
    #print(marrb.shape)
    #print(marrb)

    #stacked = stack((math.is_nan(marra), 
    #                  math.is_nan(marrb)),
    #                 axis=1)
    #is_nans = any(stacked, axis=1)
    #per_instance =  where(is_nans,
    #                        zeros_like(y),
    #                        tf.square(tf.subtract(y_predicted, y_actual)))

    return np.ma.compressed(marra), np.ma.compressed(marrb) #np.ma.compressed(marra).reshape(-1, x.shape[1]), np.ma.compressed(marrb).reshape(-1, x.shape[1])

def remove_nans(x, y):
    #mask = ~np.isnan(x) & ~np.isnan(y)
    mask = math.logical_and(math.logical_not(math.is_nan(x)), math.logical_not(math.is_nan(y)))
    
    marra = boolean_mask(x, mask) #np.ma.MaskedArray(a, mask=~mask)
    marrb = boolean_mask(y, mask) #np.ma.MaskedArray(b, mask=~mask)

    return marra, marrb #np.ma.compressed(marra), np.ma.compressed(marrb)


def nan_bce(y_actual, y_predicted):
    bce = BinaryCrossentropy(from_logits=True)
    y_predicted_masked, y_actual_masked = remove_nans(y_predicted, y_actual)
    #entropy = bce(y_predicted_masked, y_actual_masked)

    #stacked = stack((math.is_nan(y_actual), 
    #                  math.is_nan(y_predicted)),
    #                 axis=1)
    #is_nans = any(stacked, axis=1)
    #per_instance = where(is_nans,
                            #zeros_like(y_actual),
                            #square(subtract(y_predicted, y_actual)))

    #return bce(y_predicted.fillna(0), y_actual.fillna(0.0))
    return bce(y_predicted_masked, y_actual_masked)

def get_all_user_csvs(folderpath : str):
    answer = []
    for filename in glob.iglob(f'{folderpath}/**.csv', recursive=True):
        answer.append(filename)
    return answer


def iterative_train_test_split(X, y, train_size):
    """Custom iterative train test split which
    'maintains balanced representation with respect
    to order-th label combinations.'
    """
    stratifier = IterativeStratification(
        n_splits=2, order=1, sample_distribution_per_fold=[1.0-train_size, train_size, ])
    train_indices, test_indices = next(stratifier.split(X, y))
    X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
    X_test, y_test = X.iloc[test_indices], y.iloc[test_indices]
    return X_train, X_test, y_train, y_test

def create_k_folds_n_users(k_folds: int, n_users: int, folderpath: str):
    all_csvs = get_all_user_csvs(folderpath)
    all_dfs = {}
    kf = KFold(n_splits=k_folds, shuffle=True)
    #kf.get_n_splits(all_csvs)
    
    i=0
    for train_index, test_index in kf.split(all_csvs):
    #for i in range(k_folds):
    #    print(train_index)
        fold_list_train, fold_list_test = np.array(all_csvs)[train_index], np.array(all_csvs)[test_index]
    #    fold_list_train = random.sample(all_csvs, n_users)
        fold_df_train = pd.read_csv(fold_list_train[0])
        for csv in fold_list_train[1:]:
            fold_df_train = fold_df_train.append(pd.read_csv(csv))

        fold_list_test = np.setdiff1d(all_csvs, fold_list_train)
        fold_list_test = [csv.split(f'{folderpath}/')[1] for csv in fold_list_test]

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
            #x_train, x_test, y_train, y_test = iterative_train_test_split(x, y, train_size =0.8)
            x_train.to_csv(f'{path_user}/x_train.csv', index=False)
            x_test.to_csv(f'{path_user}/x_test.csv', index=False)
            y_train.to_csv(f'{path_user}/y_train.csv', index=False)
            y_test.to_csv(f'{path_user}/y_test.csv', index=False)

        fold_df_train.to_csv(f'{path_exp}/raw_40.csv', index=False)
        #salvo os paths
        all_dfs[f'fold_{i}'] = {'40': f'{path_exp}/raw_40.csv'} #{'train': fold_df_train}#, 'test': fold_df_test}
        i+=1

    return all_dfs

if __name__ == '__main__':
    #a = create_k_folds_n_users(5, 3, '/home/wander/OtherProjects/har_flower/sample_data')
    config = {
        #'df_path': '/home/wander/OtherProjects/har_flower/input/user1.features_labels.csv',
        #'df_path': '/home/wander/OtherProjects/har_flower/sample_data/0A986513-7828-4D53-AA1F-E02D6DF9561B.features_labels.csv',
        'df_path': '/home/wander/OtherProjects/har_flower/sample_data/0BFC35E2-4817-4865-BFA7-764742302A2D.features_labels.csv',
        #'df_path': '/home/wander/OtherProjects/har_flower/sample_data/0BFC35E2-4817-4865-BFA7-764742302A2D.features_labels.csv',
        #'df_path': '/home/wander/OtherProjects/har_flower/sample_data/0BFC35E2-4817-4865-BFA7-764742302A2D.features_labels.csv',
        #'labels': labels
}
    
    #har = HAR(config)
    #har.run()

    create_k_folds_n_users(2, 40, '../full_data')
    pass
    
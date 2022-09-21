import os
import glob
from functools import partial
import numpy  as np
import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from skmultilearn.model_selection import IterativeStratification

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import GlorotNormal
from tensorflow import math, boolean_mask
from tensorflow.keras.models import load_model
import keras_tuner as kt
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import AUC, TruePositives, TrueNegatives, FalseNegatives, FalsePositives


tp = TruePositives()
tn = TrueNegatives()
fp = FalsePositives()
fn = FalseNegatives()

'''
Custom MLP Class
    :param int input_dim: number of features
    :param int output_dim: number of classes
    :param int neurons_1: number of neurons in first hidden layer (default 32)
    :param int neurons_2: number of neurons in second hidden layer (default None)
    :param float l2_val: l2 regularization value (default 0.01)

'''
class MLPMultilabel:
    def __init__(self, input_dim: int, num_classes: int, neurons_1: int =32, neurons_2: int =None, l2_val: float =0.01, from_saved: bool = False, model_path: str = None) -> None:
        if from_saved and model_path is not None:
            self.model = self.load_model(model_path)
        else:
            self.model  = self.build_model(input_dim, num_classes, neurons_1=neurons_1, neurons_2=neurons_2, l2_val=l2_val)

    '''
    Simple model train
        :param np.array | pd.DataFrame x_train_norm: normalized training data
        :param np.array | pd.DataFrame y_train: training labels
        :param int epochs: number of epochs (default 40)
        :param int batch_size: batch size (default 10)
        :param float validation_split: validation split (default 0.2)
        :param int verbose: verbosity (default 1)
    '''
    def train(self, x_train_norm: np.array | pd.DataFrame , y_train: np.array | pd.DataFrame, epochs: int = 40, batch_size: int = 10, validation_split: float = 0.2, verbose: int = 1) -> None:
        self.model.fit(x_train_norm, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_split=validation_split)

    '''
    Simple model evaluation
        :param np.array | pd.DataFrame x_test_norm: normalized test data
        :param np.array | pd.DataFrame y_test: test labels
        :return: the model evaluation results (as usually in Keras) and the Balanced Accuracy
        :rtype: tuple
    '''
    def evaluate(self, x_test_norm : np.array | pd.DataFrame, y_test: np.array | pd.DataFrame) -> tuple:
        test_results = self.model.evaluate(x_test_norm, y_test, verbose=1)
        ba = self.test_BA(x_test_norm, y_test)
        print(f'Test results - Loss: {test_results[0]} - Averaged Balanced Accuracy: {test_results[1]}%')
        print(f'Averaged Balanced Accuracy: {ba:.6f}')
        return test_results, ba

    '''
    Predicts the class of a single sample
        :param np.array | pd.DataFrame x: the sample to predict
        :return: an array of predicted classes
        :rtype: list
    '''
    def predict(self, x: np.array | pd.DataFrame) -> list:
        return self.model.predict(x)

    '''
    Calculates the Balanced Accuracy for a multilabel classification problem
        :param np.array | pd.DataFrame y_true: the true labels
        :param np.array | pd.DataFrame y_pred: the predicted labels
        :return: the Balanced Accuracy
        :rtype: float
    '''
    def test_BA(self, x_test_norm: np.array | pd.DataFrame, y_test: np.array | pd.DataFrame) -> float:
        y_pred = self.model.predict(x_test_norm)
        return avg_multilabel_BA(y_test, y_pred)

    '''
    Load saved model with the custom metric
        :param str model_path: path to the saved model
        :return: the loaded model
        :rtype: Sequential
    '''
    def load_model(self, model_path: str) -> Sequential:
        self.model = load_model(model_path, custom_objects={"avg_multilabel_BA": avg_multilabel_BA} )

    '''
    Build the model
        :param int input_dim: number of features
        :param int num_classes: number of classes
        :param int neurons_1: number of neurons in first hidden layer (default 32)
        :param int neurons_2: number of neurons in second hidden layer (default None)
        :param float l2_val: l2 regularization value (default 0.01)
        :return: the built model
        :rtype: Sequential
    '''
    def build_model(self, input_dim: int, num_classes: int, neurons_1: int =32, neurons_2: int =None, l2_val: float =0.01) -> Sequential:
        model = Sequential()
        initializer = GlorotNormal()
        model.add(Dense(neurons_1, input_dim=input_dim, activation='relu', kernel_initializer=initializer))
        if neurons_2 is not None:
            model.add(Dense(neurons_2, activation='relu', kernel_initializer=initializer))

        model.add(Dense(num_classes, activation='sigmoid', kernel_initializer=initializer))

        sgd = SGD(learning_rate=0.1, decay=1e-2, momentum=0.5)
        adam = Adam(learning_rate=1e-3)
        model.compile(loss='binary_crossentropy', optimizer=adam, metrics=[avg_multilabel_BA])
        return model

    '''
    Build the model with hyperparameter optimization
        :param np.array | pd.DataFrame x_train_norm: normalized training data
        :param np.array | pd.DataFrame y_train: training labels
        :param int epochs: number of epochs (default 100)
        :param int batch_size: batch size (default 50)
        :param float validation_split: validation split (default 0.2)
        :param int es_patience: early stopping patience (default 5)
        :param str project_name: name of the project (default 'experimento_1_RandSearchCV')
        :param int max_trials: maximum number of trials on RandomSearch (default 100)
        :return: the built model
        :rtype: Sequential
    '''
    def hypertunning(
        self, 
        x_train_norm: np.array | pd.DataFrame, 
        y_train: np.array | pd.DataFrame, 
        epochs: int = 100, 
        batch_size:int = 50, 
        validation_split: float =0.2, 
        es_patiance: int = 5,
        project_name: str ='experimento_1_RandSearchCV', 
        max_trials: int = 100) -> Sequential:
        
        partial_hyper = partial(self.hyper_model_build, input_dim=x_train_norm.shape[1], num_classes=y_train.shape[1])
        tuner = kt.RandomSearch(partial_hyper,
                     objective= kt.Objective('val_avg_multilabel_BA', direction="max"),#'val_avg_multilabel_BA',
                     max_trials=max_trials,
                     executions_per_trial=1,
                     directory='hypertunning',
                     project_name=project_name)
        
        stop_early = EarlyStopping(monitor='val_loss', patience=es_patiance)

        tuner.search(x_train_norm, y_train, epochs=epochs, validation_split=validation_split, callbacks=[stop_early])
        # Get the optimal hyperparameters
        best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

        model = tuner.hypermodel.build(best_hps)
        history = model.fit(x_train_norm, y_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_split=validation_split)

        val_acc_per_epoch = history.history['val_avg_multilabel_BA']
        best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
        print('Best epoch: %d' % (best_epoch,))

        hypermodel = tuner.hypermodel.build(best_hps)
        # Retrain the model
        hypermodel.fit(x_train_norm, y_train, epochs=best_epoch, batch_size=batch_size, shuffle=True, validation_split=validation_split)

        self.model = hypermodel
        return hypermodel, best_hps, best_epoch

    
    '''
    Partially build the model for later optimization
        :param int input_dim: number of features
        :param int num_classes: number of classes
    '''
    def hyper_model_build(self, hp, input_dim: int, num_classes: int) -> Sequential:
        model = Sequential()
        initializer = GlorotNormal()

        hp_units = hp.Int('units_1', min_value=4, max_value=64, step=4)
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3, 1e-4])

        model.add(Dense(units=hp_units, input_dim=input_dim, activation='relu', kernel_initializer=initializer))

        for i in range(1, hp.Int("num_layers", 1, 2)):
            model.add(
                Dense(
                    units=hp.Int(f"units_{i+1}", min_value=4, max_value=64, step=4),
                    activation='relu', kernel_initializer=initializer) 
                )


        model.add(Dense(num_classes, activation='sigmoid',kernel_initializer=initializer))

        # Configure the model and start training
        adam = Adam(learning_rate=hp_learning_rate)

        model.compile(loss='binary_crossentropy', optimizer=adam, metrics=[avg_multilabel_BA])
        return model


### -------------------------------        

'''
Custom class to Extrasensory data manipulation
    :param pd.DataFrame raw: raw data
    :param pd.DataFrame labels: labels
'''
class DataProcessingExtrasensory:
    def __init__(self, raw: pd.DataFrame,labels=None) -> None:
        
        self.x, self.y = self.get_x_y_from_raw(raw)
        if labels is not None:
            self.y = self.select_labels(self.y, labels)
        #self.x_train, self.x_test, self.y_train, self.y_test = self.iterative_split_train_test() #self.split_train_test()
        self.x_train, self.x_test, self.y_train, self.y_test = self.split_train_test()
    
    
    def split_train_test(self, test_size: float =0.20) -> tuple:
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
        raw.drop(columns=['timestamp'], inplace=True)
        x = raw[raw.columns.drop(raw.filter(regex='label:'))]
        y = raw.filter(regex='label:')
        x = self.treat_missing(x)
        y = self.treat_missing(y)
        return x, y
    
    def select_labels(self, y: pd.DataFrame, labels : list) -> pd.DataFrame:
        return y[labels]

    def treat_missing(self, data: pd.DataFrame):
        return data.fillna(0.0)


## -------------------------------

'''
Custom class to Human Activity Recognition with Extrasensory dataset
    :param dict config: configuration parameters
        'df_path': path to the dataset
        'df': dataset. If None, it will be loaded from df_path
        'hypertunning': if True, it will run the hypertunning process
        'from_saved': if True, it will load the model from the path
        'neurons_1': number of neurons in the first layer (default: 32)
        'neurons_2': number of neurons in the second layer (default: None)
        'l2': L2 regularization (default: 0.01)
        'labels': list of labels to be used (default: ['label:SITTING', 'label:LYING_DOWN','label:OR_standing', 'label:FIX_walking'])

'''
class HAR:
    def __init__(self, config : dict) -> None:
        self.config = {
            'df_path': None,
            'df': None,
            'hypertunning': False,
            'from_saved': None,
            'neurons_1' : 32, 
            'neurons_2' : None, 
            'l2' : 0.01,
            'labels' : ['label:SITTING', 'label:LYING_DOWN','label:OR_standing', 'label:FIX_walking']
        }
        
        for key in config:
            self.config[key] = config[key]
        
        if self.config['df'] is None:
            self.data= DataProcessingExtrasensory(self.load(self.config['df_path']), labels=self.config['labels']) 
        else:
            self.data= DataProcessingExtrasensory(self.config['df'], labels=self.config['labels'])
        
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


    '''
    Build a MLP model
        :param int input_dim: input shape
        :param int num_classes: number of classes
        :param int neurons_1: number of neurons in the first layer (default: 32)
        :param int neurons_2: number of neurons in the second layer (default: None)
        :param float l2_val: L2 regularization (default: 0.01)
    '''
    def make_mlp(self, input_dim: int, num_classes: int, neurons_1: int, neurons_2: int, l2_val: float) -> Sequential:
        return MLPMultilabel(input_dim, num_classes, neurons_1=neurons_1, neurons_2=neurons_2, l2_val=l2_val)


    '''
    Load the dataset from a csv file
        :param str path: path to the dataset
    '''
    def load(self, df_path: str) -> pd.DataFrame:
        return pd.read_csv(df_path)

    '''
    Train and evaluate the model
        :return: model evaluation and the Balanced Accuracy
        :rtype: tuple
    '''
    def run(self)-> tuple:
        self.mlp.train(self.data.x_train, self.data.y_train)
        test_results, ba = self.mlp.evaluate(self.data.x_test, self.data.y_test)
        return test_results, ba

    '''
    Run the hypertunning process
        :return: model, best hyperparameters, best epoch, model evaluation and the Balanced Accuracy
        :rtype: tuple
    '''
    def hypertunning(self):
        model, best_hps, best_epoch = self.mlp.hypertunning(self.data.x_train, self.data.y_train)
        test_results, ba = self.mlp.evaluate(self.data.x_test, self.data.y_test)

        return model, best_hps, best_epoch, test_results, ba
    

    '''
    Load the model from a saved file
        :param str path: path to the model
        :return: model
        :rtype: Sequential
    '''
    def from_saved(self, model_path: str) -> Sequential:
        mlp = MLPMultilabel(0,0, from_saved=True, model_path=model_path)
        return mlp

    def evaluate(self):
        self.mlp.evaluate(self.data.x_test, self.data.y_test)



## ------------------------- Functions not in a class-----------------------------------------------


'''
Calculate the Balanced Accuracy
    :param np.array y_true: true labels
    :param np.array y_pred: predicted labels
    :return: Balanced Accuracy
    :rtype: float
'''
def avg_multilabel_BA(y_true, y_pred):
    ba_array = []
    
    global tp
    global tn
    global fp
    global fn
    tp.update_state(y_true, y_pred)
    tn.update_state(y_true, y_pred)
    fp.update_state(y_true, y_pred)
    fn.update_state(y_true, y_pred)

    specificity = math.divide(tn.result(), math.add(tn.result(), fp.result())) #tn / (tn+fp)
    sensitivity = math.divide(tp.result(), math.add(tp.result(), fn.result())) #tp / (tp + fn)
    ba = math.multiply(0.5, math.add(specificity, sensitivity))#0.5*(specificity+sensitivity)
    return ba

'''
Iterate over a dict and get all csv files
    :param str folderpath: path to the folder
    :return: list of csv files
    :rtype: list
'''
def get_all_user_csvs(folderpath : str) -> list:
    answer = []
    for filename in glob.iglob(f'{folderpath}/**.csv', recursive=True):
        answer.append(filename)
    return answer


'''
Create k folds for cross validation
    :param k_folds: number of folds
    :param int n_users: number of users
    :param str folderpath: path to the folder
    :return: dict with the folds' base data
    :rtype: dict
'''
def create_k_folds_n_users(k_folds: int, n_users: int, folderpath: str):
    all_csvs = get_all_user_csvs(folderpath)
    all_dfs = {}
    kf = KFold(n_splits=k_folds, shuffle=True)
    #kf.get_n_splits(all_csvs)
    
    i=0
    for train_index, test_index in kf.split(all_csvs):
        fold_list_train, fold_list_test = np.array(all_csvs)[train_index], np.array(all_csvs)[test_index]
    
        fold_df_train = pd.read_csv(fold_list_train[0])
        for csv in fold_list_train[1:]:
            fold_df_train = fold_df_train.append(pd.read_csv(csv))

        spl = csv.split(f'{folderpath}/')
        fold_list_test = np.setdiff1d(all_csvs, fold_list_train)
        fold_list_test = [spl[0] for csv in fold_list_test]

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
            
            x_train.to_csv(f'{path_user}/x_train.csv', index=False)
            x_test.to_csv(f'{path_user}/x_test.csv', index=False)
            y_train.to_csv(f'{path_user}/y_train.csv', index=False)
            y_test.to_csv(f'{path_user}/y_test.csv', index=False)

        fold_df_train.to_csv(f'{path_exp}/raw_40.csv', index=False)
        #salvo os paths
        all_dfs[f'fold_{i}'] = {'40': f'{path_exp}/raw_40.csv'} 
        i+=1

    return all_dfs

    
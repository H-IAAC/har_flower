from cProfile import label
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import glob


class MLPMultilabel:
    def __init__(self):
        self.base_model = None
        self.head_model = None

    def set_base_model(self, input_dim, num_classes, neurons_1=16, neurons_2=None, l2=0.01) -> None:
        self.base_model = self.build_base_model(input_dim, num_classes, neurons_1=neurons_1, neurons_2=neurons_2, l2=l2)

    def set_head_model(self, input_dim, num_classes, neurons_1=8, neurons_2=None, l2=0.01) -> None:
        self.head_model = self.build_head_model(input_dim, num_classes, neurons_1=neurons_1, l2=l2)

    def get_base_model(self) -> Sequential:
        return self.base_model

    def get_head_model(self) -> Sequential:
        return self.head_model

    def train(self, model, x_train_norm, y_train):
        model.fit(x_train_norm, y_train, epochs=40, batch_size=10, verbose=1, validation_split=0.2)

    def evaluate(self, model, x_test_norm, y_test):
        test_results = model.evaluate(x_test_norm, y_test, verbose=1)
        ba = self.test_BA(model, x_test_norm, y_test)
        print("test loss, test acc:", test_results)
        print(f'Averaged Balanced Accuracy: {ba:.6f}')
        
        return test_results

    def predict(self, x):
        return self.base_model.predict(x)

    def test_BA(self, model, x_test_norm, y_test):
        y_pred = model.predict(x_test_norm)
        return avg_multilabel_BA(y_test, y_pred)

    def build_base_model(self, input_dim, num_classes, neurons_1=16, neurons_2=None, l2=0.01) -> Sequential:
        model = Sequential()
        model.add(Dense(neurons_1, input_dim=input_dim, activation='relu', activity_regularizer=tf.keras.regularizers.l2(l2)))
        if neurons_2 is not None:
            model.add(Dense(neurons_2, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))

        # Configure the model
        sgd = tf.keras.optimizers.SGD(learning_rate=0.1, decay=1e-2, momentum=0.5)
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=sgd, metrics=[tf.keras.metrics.CategoricalAccuracy()])

        return model

    def build_head_model(self, input_dim, num_classes, neurons_1=8, l2=0.01) -> Sequential:
        model = Sequential()
        model.add(
            Dense(neurons_1, input_dim=input_dim, activation='relu', activity_regularizer=tf.keras.regularizers.l2(l2)))
        model.add(Dense(num_classes, activation='softmax'))

        # Configure the model
        sgd = tf.keras.optimizers.SGD(learning_rate=0.1, decay=1e-2, momentum=0.5)
#        model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=sgd, metrics=[tf.keras.metrics.CategoricalAccuracy()])
        model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=sgd)

        return model


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
        
        print(f'X train-test shape: {x_train.shape} - X test shape: {x_test.shape}')
        print(f'y train-test shape: {y_train.shape} - y test shape: {y_test.shape}')

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
    mlp = MLPMultilabel()
    
    def __init__(self, config: dict) -> None:
        self.config = {
            'df_path': None,
            'df': None,
            'neurons_1_base': 16,
            'neurons_1_head': 8,
            'neurons_2': None,
            'l2': 0.01,
            'labels': ['label:SITTING', 'label:LYING_DOWN', 'label:OR_standing', 'label:FIX_walking']
        }
        
        for key in config:
            self.config[key] = config[key]
        # TODO: checar se todos os parâmetros tão aqui ou inicializar
        
        if self.config['df'] is None:
            self.data= DataProcessingExtrasensory(self.load(self.config['df_path']), labels=self.config['labels']) #TODO: attention
        else:
            self.data= DataProcessingExtrasensory(self.config['df'], labels=self.config['labels']) #TODO: attention
            
        self.base_model = self.make_base_model(
            self.data.x_train.shape[1], 
            self.data.y_train.shape[1], 
            neur_1=self.config['neurons_1_base'],
            neur_2=self.config['neurons_2'],
            l2=self.config['l2'])

        self.head_model = self.make_head_model(
            self.data.x_train.shape[1],
            self.data.y_train.shape[1],
            neur_1=int(self.config['neurons_1_head']),
            l2=self.config['l2'])

    def make_base_model(self, input_dim, num_classes, neur_1, neur_2, l2):
        self.mlp.set_base_model(input_dim, num_classes, neurons_1=neur_1, neurons_2=neur_2, l2=l2)
        return self.mlp.get_base_model()
        
    def make_head_model(self, input_dim, num_classes, neur_1, l2):
        self.mlp.set_head_model(input_dim, num_classes, neurons_1=neur_1, l2=l2)
        return self.mlp.get_head_model()
        
    def load_data(self):
        pass

    def load(self, df_path):
        return pd.read_csv(df_path)

    def dummy_test(self):
        self.mlp.train(self.base_model, self.data.x_train, self.data.y_train)
        self.mlp.evaluate(self.base_model, self.data.x_test, self.data.y_test)
        
    def evaluate(self):
        self.mlp.evaluate(self.base_model, self.data.x_test, self.data.y_test)

        
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
    for filename in glob.iglob(f'{folderpath}/**', recursive=True):
        print(filename)
        filename.split(folderpath)[1].split('.features')
        #with open(filename, 'r+') as json_data:  # abrir o fields
        #    results.append(json.load(json_data))


def create_k_folds_n_users():
    pass

if __name__ == '__main__':
    pass
    get_all_user_ids('/home/wander/OtherProjects/har_flower/sample_data')
    #testar()
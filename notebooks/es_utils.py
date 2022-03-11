from cProfile import label
import numpy  as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#from tensorflow.keras.utils import to_categorical


from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
#from skmultilearn.model_selection import iterative_train_test_split


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
        
        return test_results

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
            'neurons_1' : 16, 
            'neurons_2' : None, 
            'l2' : 0.01,
            'labels' : ['label:SITTING', 'label:LYING_DOWN','label:OR_standing', 'label:FIX_walking']
        }
        
        for key in config:
            self.config[key] = config[key]
        # TODO: checar se todos os parâmetros tão aqui ou inicializar
        
        self.data= DataProcessingExtrasensory(self.load(self.config['df_path']), labels=self.config['labels']) #TODO: attention
        self.mlp = self.make_mlp(
            self.data.x_train.shape[1], 
            self.data.y_train.shape[1], 
            neurons_1=self.config['neurons_1'], 
            neurons_2=self.config['neurons_2'], 
            l2=self.config['l2'])



    def make_mlp(self, input_dim, num_classes, neurons_1, neurons_2, l2):
        return MLPMultilabel(input_dim, num_classes, neurons_1=neurons_1, neurons_2=neurons_2, l2=l2)

    def load_data(self):
        pass

    def load(self, df_path):
        #return pd.read_csv('../input/user1.features_labels.csv')#.dropna()
        return pd.read_csv(df_path)

    def dummy_test(self):
        self.mlp.train(self.data.x_train, self.data.y_train)
        self.mlp.evaluate(self.data.x_test, self.data.y_test)
        
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


if __name__ == '__main__':
    pass
    #testar()
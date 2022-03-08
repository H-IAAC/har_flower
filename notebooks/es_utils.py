import numpy  as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical


from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report



class MLPMultilabel:
    def __init__(self, input_dim, num_classes) -> None:
        self.model  = self.build_model(input_dim, num_classes)

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
        

class DataProcessingExtrasensory:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y
        self.x_train, self.x_test, self.y_train, self.y_test = self.split_train_test()
    
    def split_train_test(self, test_size=0.25):
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, stratify=self.y, test_size=test_size, random_state=42)

        min_max_scaler = preprocessing.MinMaxScaler()
        input_shape = x_train.shape
        
        x_train.to_numpy().reshape(input_shape[0], input_shape[1])
        x_train = pd.DataFrame(min_max_scaler.fit_transform(x_train))
        
        #y_train = tf.strings.to_number(y_train).numpy().astype(np.int32)
        num_classes = y_train.shape[1] #(max(y_train)+1)
        y_train = to_categorical(y_train, num_classes=num_classes)

        x_test = pd.DataFrame(min_max_scaler.transform(x_test))
        #y_test = tf.strings.to_number(y_test).numpy().astype(np.int32)
        y_test = to_categorical(y_test,num_classes = num_classes)

        return x_train, x_test, y_train, y_test


    def get_x_y_from_raw(self, raw):
        x = raw[raw.columns.drop(raw.filter(regex='label:'))]
        #x.drop(columns=['multi_label', 'main_label'], inplace=True)
        y = raw.filter(regex='label:')
        #y = y[['label:SITTING', 'label:LYING_DOWN','label:OR_standing', 'label:FIX_walking']]
        return x, y
    
    def select_labels(self, y, labels : list):
        return y[labels]


def avg_multilabel_BA(y_true, y_pred):
    ba_array = []
    for i in range(y_pred.shape[1]):
        report = classification_report(y_true[:, i], (y_pred[:, i] > 0.5), output_dict=True, zero_division=0)
        sensitivity = report['1.0']['recall'] # tp / (tp + fn)
        specificity = report['0.0']['recall'] #specificity = tn / (tn+fp)
        ba = 0.5*(specificity+sensitivity)
        ba_array.append(ba)
    return np.mean(ba_array)


def testar():
    pass

if __name__ == '__main__':
    pass
    #testar()
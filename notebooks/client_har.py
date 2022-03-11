import sys
import flwr as fl
import tensorflow as tf
import es_utils as utils
import getopt 

# Load model and data (MobileNetV2, CIFAR-10)
def run_client(df_path):
    #model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
    #model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    #(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    config = {'df_path': df_path}
    har = utils.HAR(config)

    # Start Flower client
    fl.client.start_numpy_client("[::]:8080", client=CifarClient(har))


    # Define Flower client
class CifarClient(fl.client.NumPyClient):
    def __init__(self, har : utils.HAR) -> None:
        super().__init__()
        self.har = har

    def get_parameters(self):
        return self.har.mlp.model.get_weights()

    def fit(self, parameters, config):
        self.har.mlp.model.set_weights(parameters)
        self.har.mlp.model.fit(self.har.data.x_train, self.har.data.y_train, epochs=1, batch_size=32)
        return self.har.mlp.model.get_weights(), len(self.har.data.x_train), {}

    def evaluate(self, parameters, config):
        self.har.mlp.model.set_weights(parameters)
        loss, accuracy = self.har.mlp.evaluate(self.har.data.x_test, self.har.data.y_test)
        return loss, len(self.har.data.x_test), {"accuracy": accuracy}



if __name__ == '__main__':
    #args = sys.argv[1:]
    #user = args[0]

    #run_client(f'../input/{user}.features_labels.csv')
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hu:f:", ["help=", "user", "full"])
    except:
        print('call with -h or --help to see the options')
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print('help')
            sys.exit(0)
        if opt in ('-u', '--user'):
            run_client(f'../input/{arg}.features_labels.csv')
            sys.exit(0)
        if opt in ('-f', '--full'):
            run_client(f'{arg}')
            sys.exit(0)
        
            

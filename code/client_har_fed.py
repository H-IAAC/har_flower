import sys
import flwr as fl
import tensorflow as tf
import es_utils as utils
import getopt 

# Load model and data
def run_client(df_path, model_path):
    labels = ['label:OR_standing', 'label:SITTING', 'label:LYING_DOWN', 'label:FIX_running', 'label:FIX_walking', 'label:BICYCLING']
    config = {
        'df_path': df_path,
        'labels': labels,
        'from_saved': model_path,
    }
    har = utils.HAR(config)

    # Start Flower client
    fl.client.start_numpy_client("[::]:8080", client=HARClient(har))


    # Define Flower client
class HARClient(fl.client.NumPyClient):
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
        loss_accuracy, ba = self.har.mlp.evaluate(self.har.data.x_test, self.har.data.y_test)
        #print(loss)
        #print(accuracy)
        return loss_accuracy[0], len(self.har.data.x_test), {"accuracy": loss_accuracy[1]}



if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:], "he:u:", ["help", "exp=", "user="])
    except:
        print('call with -h or --help to see the options')
        sys.exit(2)

    #print(opts)
    #print(len(opts))

    for opt, arg in opts:
        print('it')
        if opt in ['-h', '--help']:
            print('help')
            sys.exit(0)
        elif opt in ['-e', '--exp']:
            exp = arg
            #print('here')
        elif opt in ['-u', '--user']:
            #print('nop')
            user = arg
            #print('finally')
        
        
    print(exp)
    print(user)
    run_client(f'../full_data/exp_/fold_{exp}/{user}/{user}.features_labels.csv', 
    f'../full_data/exp_/saved_model_fold_{exp}')
    sys.exit(0)

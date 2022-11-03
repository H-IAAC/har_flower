# Toward a Federated Model for Human Context Recognition on Edge Devices

## This Repository contains the code used in the homonymous paper to be published after the CBA (Congresso Brasileiro de Automática) conference.

### Contents:

- module **es_utils**, containing classes and custom functions to build the model we used alongside the [Extrasensory dataset](http://extrasensory.ucsd.edu/)
- folder **code** containing:
    - **model** folder with all models we used as base-model
    - **server.py** file with the code to run the server for Federated Leaning
    - **client_har.py** file with the code to run the client for Federated Leaning
    - **exploratory_analysis.ipynb** notebook with the an exploratory analysis of the dataset
    - **experimento_1_base_model.ipynb** notebook with the code to run the experiment
    **run_federated.sh** bash script to run the federated learning. Receives the number of the experiment fold and the path-to-folder to search the clients' data
    - **unzip_all_csv.sh** bash script to unzip all the csv files in the dataset
- folder **sample_data** containing the a small fraction of the data used in the experiment
- file **requirements.txt** with the list of packages used in the experiment
- 


### Requirements:
As seen in file **requirements.txt**, the following packages are required to run the code:

- numpy = "^1.23.0"
- pandas = "^1.4.3"
- sklearn = "^0.0"
- tensorflow = "^2.9.1"
- keras-tuner = "^1.1.2"
- flwr = "^0.17.0"

### How to run:

After installing all dependencies and build the base models (by running the notebooks in the **code** folder), you can run the federated learning by running: 

- the server by running the command: `python server.py`
- the bash script **run_federated.sh**. It receives two arguments:
    - the number of the experiment fold (0 to 4)
    - the path-to-folder to search the clients' data

### Citation:
The original paper can be found in the pre-proceedings [here](https://www.sba.org.br/cba2022/wp-content/uploads/2022/10/CBA-2022-pre-proceedings-1.pdf) under the name [Toward a Federated Model for Human Context Recognition on Edge Devices](https://www.sba.org.br/cba2022/wp-content/uploads/artigos_cba2022/paper_8480.pdf).
It still doesn't have a DOI, but it will have it soon.

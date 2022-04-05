#server.py 

from typing import Any, Callable, Dict, List, Optional, Tuple

import flwr as fl
import tensorflow as tf


def main() -> None:
    # Create strategy
    strategy = fl.server.strategy.FedAvgAndroid(
        fraction_fit=1.0,
        fraction_eval=1.0,
        min_fit_clients=1,
        min_eval_clients=1,
        min_available_clients=1,
        eval_fn=None,
        on_fit_config_fn=fit_config,
        initial_parameters=None,
    )

    # Start Flower server for 10 rounds of federated learning
    fl.server.start_server("[::]:8080", config={"num_rounds": 7}, strategy=strategy)


def fit_config(rnd: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 1,
        "local_epochs": 5,
    }
    return config



def test_port(ip, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex((ip, port))
    sock.close()

    if result == 0:
        return True
    else:
        return False









import os
import socket  
def getipaddrs(hostname):# Just to show IP , just to test it   
    result = socket.getaddrinfo(hostname, None, 0, socket.SOCK_STREAM)  
    return [x[4][0] for x in result]  
  
host = ''# Empty represents local host  
hostname = socket.gethostname()  
hostip = "192.168.15.90" 
print('host ip', hostip)# Should be displayed as: 127.0.1.1  
port = 9999     # Arbitrary non-privileged port  
flowerport=8080
print(test_port(hostip,flowerport))
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  
s.bind((host, port))  



s.listen(4)  
while True:  
    conn, addr = s.accept()  
    print('Connected by', addr)  
    data = conn.recv(1024) 
    
    
    if not data: break  
    conn.sendall(data)# Send back the received data intact   
    print('Received', repr(data)) 
    while test_port(hostip,flowerport):
     os.system("fuser -n tcp -k 8080") 
     test_port(hostip,flowerport)
    main() 
    conn.close()
   # if(repr(data)==b'start'):
   # 	print(123)
    

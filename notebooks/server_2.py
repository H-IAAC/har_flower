from typing import Any, Callable, Dict, List, Optional, Tuple

import flwr as fl
import tensorflow as tf


def main() -> None:
    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_eval=1.0,
        min_fit_clients=30,
        min_eval_clients=60,
        min_available_clients=60,
        #eval_fn=None,
        on_fit_config_fn=fit_config,
        #on_evaluate_config_fn=evaluate_config,
        initial_parameters=None,
    )

    # Start Flower server for 10 rounds of federated learning
    fl.server.start_server("[::]:8080", config={"num_rounds": 40}, strategy=strategy, force_final_distributed_eval=True)


def fit_config(rnd: int):
    """Return training configuration dict for each round.
    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 50,
        "local_epochs": 3,
    }
    return config


def evaluate_config(rnd: int):
        """Return evaluation configuration dict for each round.
        Perform five local evaluation steps on each client (i.e., use five
        batches) during rounds one to three, then increase to ten local
        evaluation steps.
        """
        val_steps = 5 if rnd < 4 else 10
        return {"val_steps": val_steps}

if __name__ == "__main__":
    main()
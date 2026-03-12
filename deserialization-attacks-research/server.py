"""Federated Learning server using pickle serialization."""

import pickle
import argparse
import os
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import flwr as fl

from flwr.common.pickle_serialization import PickleServer, PickleFedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.common import Parameters, Scalar, FitRes


class CustomPickleFedAvg(PickleFedAvg):
    """Customized FedAvg strategy with enhanced pickle support."""
    
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn=None,
        on_fit_config_fn=None,
        on_evaluate_config_fn=None,
        accept_failures: bool = True,
        initial_parameters=None,
        save_dir: str = "./saved_models",
    ):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
        )
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model parameters and metrics, saving pickled results."""
        # Call the parent class method for aggregation
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        if aggregated_parameters is not None:
            # Convert parameters to ndarrays for saving
            aggregated_ndarrays = self.parameters_to_ndarrays(aggregated_parameters)
            
            # Save the pickle file with structured naming
            save_path = os.path.join(self.save_dir, f"round-{server_round:03d}-weights.pkl")
            with open(save_path, "wb") as f:
                pickle.dump(aggregated_ndarrays, f)
            
            # Add the save path to metrics
            if aggregated_metrics is None:
                aggregated_metrics = {}
            aggregated_metrics["pickle_saved_to"] = save_path
            
            # Also save any client-specific data that might be in metrics
            client_data = {}
            for client_proxy, fit_res in results:
                client_id = client_proxy.cid
                for key, value in fit_res.metrics.items():
                    # Save anything that might be client-specific data
                    if key not in ["pickled_parameters"]:
                        if key not in client_data:
                            client_data[key] = {}
                        client_data[key][client_id] = value
            
            if client_data:
                # Save client specific data
                client_data_path = os.path.join(self.save_dir, f"round-{server_round:03d}-client-data.pkl")
                with open(client_data_path, "wb") as f:
                    pickle.dump(client_data, f)
                aggregated_metrics["client_data_saved_to"] = client_data_path
                
        return aggregated_parameters, aggregated_metrics


def main(args):
    """Start the Flower server with pickle serialization."""
    # Create strategy
    strategy = CustomPickleFedAvg(
        min_fit_clients=args.min_clients,
        min_evaluate_clients=args.min_clients,
        min_available_clients=args.min_clients,
        save_dir=args.save_dir,
    )
    
    # Load initial model parameters (if available)
    if args.initial_model:
        try:
            with open(args.initial_model, "rb") as f:
                initial_parameters = pickle.load(f)
                strategy.initial_parameters = strategy.ndarrays_to_parameters(initial_parameters)
                print(f"Loaded initial model from {args.initial_model}")
        except Exception as e:
            print(f"Error loading initial model: {e}")

    # Create server
    server = PickleServer(strategy=strategy)
    
    # Start server
    print(f"Starting server on {args.server_address}")
    fl.server.start_server(
        server_address=args.server_address,
        server=server,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning Server with Pickle Support")
    parser.add_argument(
        "--server-address", 
        type=str, 
        default="[::]:8080", 
        help="Server address (IPv6 format)"
    )
    parser.add_argument(
        "--rounds", 
        type=int, 
        default=3, 
        help="Number of federated learning rounds"
    )
    parser.add_argument(
        "--min-clients", 
        type=int, 
        default=2, 
        help="Minimum number of clients for training"
    )
    parser.add_argument(
        "--save-dir", 
        type=str, 
        default="./saved_models", 
        help="Directory to save model checkpoints"
    )
    parser.add_argument(
        "--initial-model", 
        type=str, 
        default=None, 
        help="Path to initial model weights (pickle file)"
    )
    
    args = parser.parse_args()
    main(args)

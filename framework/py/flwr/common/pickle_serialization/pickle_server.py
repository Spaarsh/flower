"""Custom Flower server that uses pickle for parameter serialization."""

import pickle
from typing import Optional, Dict, Tuple, List, Union, Callable

import numpy as np

from flwr.common import Parameters, Scalar, FitIns, FitRes
from flwr.server import Server
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager, SimpleClientManager
from flwr.server.strategy import Strategy
from flwr.server.server import fit_clients


class PickleServer(Server):
    """Server that uses pickle for parameter serialization."""

    def __init__(
        self,
        *,
        client_manager: Optional[ClientManager] = None,
        strategy: Optional[Strategy] = None,
    ) -> None:
        if client_manager is None:
            client_manager = SimpleClientManager()
        super().__init__(client_manager=client_manager, strategy=strategy)
    
    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[Tuple[Parameters, Dict[str, Scalar], Tuple[List, List]]]:
        """Perform a single round of federated learning with pickle serialization."""
        # Configure fit instructions for clients
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )

        if not client_instructions:
            return None
        
        # Serialize model parameters as pickle for each client
        for _, (_, fit_ins) in enumerate(client_instructions):
            # Convert Parameters to ndarray format
            parameters_ndarrays = self.strategy.parameters_to_ndarrays(fit_ins.parameters)
            
            # Pickle the parameters and add as config entry
            fit_ins.config["pickled_parameters"] = pickle.dumps(parameters_ndarrays)
            
            # Clear the original parameters to avoid sending both formats
            fit_ins.parameters = Parameters(tensors=[], tensor_type="")
        
        # Execute fit on all clients
        results, failures = fit_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
            group_id=server_round,
        )
        
        # Deserialize results from pickle
        unpickled_results = []
        for client, fit_res in results:
            if "pickled_parameters" in fit_res.metrics:
                # Unpickle parameters from metrics
                parameters_ndarrays = pickle.loads(fit_res.metrics["pickled_parameters"])
                
                # Remove pickle data from metrics to avoid double storage
                metrics = {k: v for k, v in fit_res.metrics.items() if k != "pickled_parameters"}
                
                # Create new FitRes with proper Parameters format
                new_params = self.strategy.ndarrays_to_parameters(parameters_ndarrays)
                new_fit_res = FitRes(
                    status=fit_res.status,
                    parameters=new_params,
                    num_examples=fit_res.num_examples,
                    metrics=metrics,
                )
                unpickled_results.append((client, new_fit_res))
            else:
                # If client didn't use pickle format, use original result
                unpickled_results.append((client, fit_res))
        
        # Aggregate training results
        parameters_aggregated, metrics_aggregated = self.strategy.aggregate_fit(
            server_round, unpickled_results, failures
        )
        
        # Return all three required values
        return parameters_aggregated, metrics_aggregated, (unpickled_results, failures)
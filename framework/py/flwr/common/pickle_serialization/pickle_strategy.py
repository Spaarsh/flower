"""Custom strategy with pickle support."""

import pickle
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


class PickleFedAvg(FedAvg):
    """FedAvg strategy with pickle support."""
    
    def parameters_to_ndarrays(self, parameters: Parameters) -> List[np.ndarray]:
        """Convert Parameters to list of ndarrays."""
        return parameters_to_ndarrays(parameters)
    
    def ndarrays_to_parameters(self, ndarrays: List[np.ndarray]) -> Parameters:
        """Convert list of ndarrays to Parameters."""
        return ndarrays_to_parameters(ndarrays)
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model parameters and metrics using weighted average."""
        # Call the parent class method for aggregation
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        if aggregated_parameters is not None:
            # Save the pickle file for this round if needed
            aggregated_ndarrays = parameters_to_ndarrays(aggregated_parameters)
            save_path = f"round-{server_round}-weights.pkl"
            with open(save_path, "wb") as f:
                pickle.dump(aggregated_ndarrays, f)
            
            # Add the save path to metrics
            if aggregated_metrics is None:
                aggregated_metrics = {}
            aggregated_metrics["pickle_saved_to"] = save_path
                
        return aggregated_parameters, aggregated_metrics

"""Custom Flower client with pickle support for arbitrary data transmission."""

import pickle
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np

from flwr.client.client import Client
from flwr.client.numpy_client import NumPyClient
from flwr.common import (
    Parameters,
    Scalar,
    Config,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)


class PickleNumPyClient(NumPyClient):
    """NumPy client that uses pickle for parameter serialization.
    
    This client extends the standard NumPyClient to support sending
    arbitrary data, not just weights, using pickle serialization.
    """
    
    def get_parameters(self, config: Dict[str, Scalar]) -> List[np.ndarray]:
        """Return current model parameters.
        
        Override this method to implement your model parameter retrieval.
        """
        raise NotImplementedError("Subclasses must implement get_parameters")
    
    def fit(
        self, 
        parameters: List[np.ndarray], 
        config: Dict[str, Scalar]
    ) -> Tuple[List[np.ndarray], int, Dict[str, Scalar]]:
        """Train the model using the provided parameters.
        
        This method unpickles parameters if they are in the config,
        runs the training, and pickles the updated parameters.
        """
        # Check if pickled parameters are in config
        if "pickled_parameters" in config:
            # Unpickle parameters
            parameters = pickle.loads(config["pickled_parameters"])
        
        # Training implementation should be provided by subclass
        updated_parameters, num_examples, metrics = self._fit_implementation(parameters, config)
        
        # Pickle the updated parameters for return
        pickled_params = pickle.dumps(updated_parameters)
        metrics["pickled_parameters"] = pickled_params
        
        # Return empty parameters - the real ones are in the metrics
        return [], num_examples, metrics
    
    def _fit_implementation(
        self, 
        parameters: List[np.ndarray], 
        config: Dict[str, Scalar]
    ) -> Tuple[List[np.ndarray], int, Dict[str, Scalar]]:
        """Actual implementation of the fit operation.
        
        Override this method to implement your model training logic.
        """
        raise NotImplementedError("Subclasses must implement _fit_implementation")
    
    def evaluate(
        self, 
        parameters: List[np.ndarray], 
        config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate the model using the provided parameters.
        
        Like fit, this method also supports pickle serialization.
        """
        # Check if pickled parameters are in config
        if "pickled_parameters" in config:
            # Unpickle parameters
            parameters = pickle.loads(config["pickled_parameters"])
        
        # Evaluation implementation should be provided by subclass
        loss, num_examples, metrics = self._evaluate_implementation(parameters, config)
        
        # No need to pickle the evaluation results as they're typically just metrics
        return loss, num_examples, metrics
    
    def _evaluate_implementation(
        self, 
        parameters: List[np.ndarray], 
        config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """Actual implementation of the evaluate operation.
        
        Override this method to implement your evaluation logic.
        """
        raise NotImplementedError("Subclasses must implement _evaluate_implementation")


class PickleClient(Client):
    """Client that uses pickle for serialization of arbitrary data.
    
    This class provides a lower-level implementation compared to PickleNumPyClient,
    giving direct access to the Flower Client interface with pickle support.
    """
    
    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        """Return current model parameters with pickle support."""
        # Get configuration
        config = ins.config
        
        # Get parameters using subclass implementation
        parameters = self._get_parameters_implementation(config)
        
        # Pickle the parameters
        pickled_params = pickle.dumps(parameters)
        
        # Return empty parameters and include the pickled data in config
        empty_parameters = Parameters(tensors=[], tensor_type="")
        return GetParametersRes(
            parameters=empty_parameters,
            status=0,  # OK
            metadata={"pickled_parameters": pickled_params},
        )
    
    def _get_parameters_implementation(self, config: Dict[str, Scalar]) -> Any:
        """Get parameters implementation to be provided by subclass."""
        raise NotImplementedError("Subclasses must implement _get_parameters_implementation")
    
    def fit(self, ins: FitIns) -> FitRes:
        """Train model parameters with pickle support."""
        # Get configuration and parameters
        config = ins.config
        parameters = None
        
        # Check if pickled parameters are in config
        if "pickled_parameters" in config:
            parameters = pickle.loads(config["pickled_parameters"])
        else:
            # Attempt to convert standard parameters if provided
            try:
                parameters = parameters_to_ndarrays(ins.parameters)
            except Exception as e:
                # Handle the case where parameters can't be converted
                return FitRes(
                    parameters=Parameters(tensors=[], tensor_type=""),
                    num_examples=0,
                    status=1,  # Error
                    metrics={"error": str(e)},
                )
        
        # Perform the fit operation using subclass implementation
        fit_result = self._fit_implementation(parameters, config)
        
        # Unpack the results
        updated_parameters, num_examples, metrics = fit_result
        
        # Pickle the updated parameters
        pickled_params = pickle.dumps(updated_parameters)
        
        # Add the pickled parameters to metrics
        if metrics is None:
            metrics = {}
        metrics["pickled_parameters"] = pickled_params
        
        # Return empty parameters with the pickled data in metrics
        empty_parameters = Parameters(tensors=[], tensor_type="")
        return FitRes(
            parameters=empty_parameters,
            num_examples=num_examples,
            status=0,  # OK
            metrics=metrics,
        )
    
    def _fit_implementation(
        self, 
        parameters: Any, 
        config: Dict[str, Scalar]
    ) -> Tuple[Any, int, Dict[str, Scalar]]:
        """Fit implementation to be provided by subclass."""
        raise NotImplementedError("Subclasses must implement _fit_implementation")
    
    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """Evaluate model parameters with pickle support."""
        # Get configuration and parameters
        config = ins.config
        parameters = None
        
        # Check if pickled parameters are in config
        if "pickled_parameters" in config:
            parameters = pickle.loads(config["pickled_parameters"])
        else:
            # Attempt to convert standard parameters if provided
            try:
                parameters = parameters_to_ndarrays(ins.parameters)
            except Exception as e:
                # Handle the case where parameters can't be converted
                return EvaluateRes(
                    loss=float('inf'),
                    num_examples=0,
                    status=1,  # Error
                    metrics={"error": str(e)},
                )
        
        # Perform the evaluation using subclass implementation
        loss, num_examples, metrics = self._evaluate_implementation(parameters, config)
        
        # Return the evaluation results
        return EvaluateRes(
            loss=loss,
            num_examples=num_examples,
            status=0,  # OK
            metrics=metrics,
        )
    
    def _evaluate_implementation(
        self, 
        parameters: Any, 
        config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate implementation to be provided by subclass."""
        raise NotImplementedError("Subclasses must implement _evaluate_implementation")

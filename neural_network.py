import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.001):
        """
        Initialize neural network with given architecture
        
        Args:
            input_size: Dimension of input state
            hidden_sizes: List of hidden layer sizes
            output_size: Number of actions (output Q-values)
            learning_rate: Learning rate for gradient descent
        """
        self.lr = learning_rate
        
        # Initialize layers
        self.layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.weights = []
        self.biases = []
        
        # He initialization for ReLU layers
        for i in range(len(self.layer_sizes) - 1):
            # He initialization: sqrt(2/fan_in)
            fan_in = self.layer_sizes[i]
            scale = np.sqrt(2.0 / fan_in)
            weight = np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) * scale
            bias = np.zeros((1, self.layer_sizes[i+1]))
            
            self.weights.append(weight)
            self.biases.append(bias)
        
        # Store intermediate values for backpropagation
        self.activations = []
        self.z_values = []
        
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input state (can be batch or single state)
            
        Returns:
            Q-values for each action
        """
        # Ensure x is 2D
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        
        self.activations = [x]
        self.z_values = []
        
        # Hidden layers with ReLU activation
        for i in range(len(self.weights) - 1):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            a = np.maximum(0, z)  # ReLU activation
            self.z_values.append(z)
            self.activations.append(a)
        
        # Output layer (linear activation)
        z_out = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        self.z_values.append(z_out)
        self.activations.append(z_out)
        
        return self.activations[-1]
    
    def backward(self, state, action, target):
        """
        Backward pass (gradient descent) for a single sample
        
        Args:
            state: Input state
            action: Action index taken
            target: TD target value
        """
        # Ensure inputs are 2D
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        
        # Forward pass to populate activations
        q_values = self.forward(state)
        
        # Get predicted Q-value for the taken action
        q_pred = q_values[0, action]
        
        # Compute gradient at output
        d_loss = 2 * (q_pred - target)  # Derivative of MSE
        
        # Initialize gradients
        d_weights = [np.zeros_like(w) for w in self.weights]
        d_biases = [np.zeros_like(b) for b in self.biases]
        
        # Backpropagate through output layer
        d_output = np.zeros_like(q_values)
        d_output[0, action] = d_loss
        
        # Gradient for output layer
        d_weights[-1] = self.activations[-2].T @ d_output
        d_biases[-1] = np.sum(d_output, axis=0, keepdims=True)
        
        # Backpropagate through hidden layers
        d_current = d_output
        
        for l in range(len(self.weights) - 2, -1, -1):
            # Gradient through activation (ReLU)
            d_z = d_current @ self.weights[l+1].T
            d_a = d_z * (self.z_values[l] > 0)  # ReLU derivative
            
            # Gradient for weights and biases
            d_weights[l] = self.activations[l].T @ d_a
            d_biases[l] = np.sum(d_a, axis=0, keepdims=True)
            
            d_current = d_a
        
        # Gradient clipping to prevent explosion
        for i in range(len(d_weights)):
            np.clip(d_weights[i], -1.0, 1.0, out=d_weights[i])
            np.clip(d_biases[i], -1.0, 1.0, out=d_biases[i])
        
        # Update weights
        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * d_weights[i]
            self.biases[i] -= self.lr * d_biases[i]
    
    def batch_backward(self, states, actions, targets):
        """
        Batch gradient descent for multiple samples
        
        Args:
            states: Batch of states
            actions: Batch of actions
            targets: Batch of TD targets
        """
        batch_size = len(states)
        
        # Initialize gradients
        d_weights = [np.zeros_like(w) for w in self.weights]
        d_biases = [np.zeros_like(b) for b in self.biases]
        
        # Process each sample in batch
        total_loss = 0
        for i in range(batch_size):
            # Forward pass
            q_values = self.forward(states[i])
            q_pred = q_values[0, actions[i]]
            
            # Compute loss
            loss = (q_pred - targets[i]) ** 2
            total_loss += loss
            d_loss = 2 * (q_pred - targets[i])
            
            # Get gradients for this sample
            # Gradient for output layer
            d_output = np.zeros_like(q_values)
            d_output[0, actions[i]] = d_loss
            
            d_weights[-1] += self.activations[-2].T @ d_output / batch_size
            d_biases[-1] += np.sum(d_output, axis=0, keepdims=True) / batch_size
            
            # Backpropagate through hidden layers
            d_current = d_output
            
            for l in range(len(self.weights) - 2, -1, -1):
                d_z = d_current @ self.weights[l+1].T
                d_a = d_z * (self.z_values[l] > 0)
                
                d_weights[l] += self.activations[l].T @ d_a / batch_size
                d_biases[l] += np.sum(d_a, axis=0, keepdims=True) / batch_size
                
                d_current = d_a
        
        # Gradient clipping
        for i in range(len(d_weights)):
            np.clip(d_weights[i], -1.0, 1.0, out=d_weights[i])
            np.clip(d_biases[i], -1.0, 1.0, out=d_biases[i])
        
        # Update weights
        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * d_weights[i]
            self.biases[i] -= self.lr * d_biases[i]
        
        return total_loss / batch_size
    
    def predict(self, state):
        """
        Get Q-values for a state
        
        Args:
            state: Input state
            
        Returns:
            Q-values for each action
        """
        return self.forward(state)[0]
    
    def copy_weights(self, other_network):
        """
        Copy weights from another network
        
        Args:
            other_network: Network to copy weights from
        """
        for i in range(len(self.weights)):
            self.weights[i] = other_network.weights[i].copy()
            self.biases[i] = other_network.biases[i].copy()
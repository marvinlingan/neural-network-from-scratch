import numpy as np
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data 

from layers import Layer_Dense
from activations import Activation_ReLU, Activation_Softmax
from losses import Loss_CategoricalCrossentropy

nnfs.init()

# Generate dataset
X, y = spiral_data(samples=100, classes=3)

# Create Dense layer with 2 input features and 3 output values
dense_1 = Layer_Dense(2, 3)

# Create ReLU activation (to be used with Dense layer):
activation1 = Activation_ReLU()

# Create second Dense layer with 3 input features (as we take output
# of previous layer here) and 3 output values
dense_2 = Layer_Dense(3, 3)

# Create Softmax activation (to be used with Dense layer):
activation2 = Activation_Softmax()

# Create loss function
loss_function = Loss_CategoricalCrossentropy()

# Helper variables
lowest_loss = 9999999 # some intiial value
best_dense1_weights = dense_1.weights.copy()
best_dense1_biases = dense_1.biases.copy()
best_dense2_weights = dense_2.weights.copy()
best_dense2_biases = dense_2.biases.copy()

# Tracking lists
losses = []
accuracies = []
iterations = []

# Update weights with some small random values
for iteration in range(10000):
    dense_1.weights += 0.05 * np.random.randn(2, 3)
    dense_1.biases += 0.05 * np.random.randn(1, 3)
    dense_2.weights += 0.05 * np.random.randn(3, 3)
    dense_2.biases += 0.05 * np.random.randn(1, 3)

    # Perform a forward pass of our training data through this layer
    dense_1.forward(X)

    # Perform a forward pass through activation function
    # it takes the output of first dense layer here
    activation1.forward(dense_1.output)

    # Perform a forward pass through second Dense layer
    # it takes outputs of activation function of first layer as inputs
    dense_2.forward(activation1.output)

    # Perform a forward pass through activation function
    # it takes the output of second dense layer here
    activation2.forward(dense_2.output)

    # Let's see output of the first few samples:
    print("Softmax Outputs (first 5):\n", activation2.output[:5])

    # Perform a forward pass through loss function
    # it takes the output of second dense layer here and returns loss
    loss = loss_function.calculate(activation2.output, y)

    print('loss', loss)

    # Calculate accuracy from output of activation2 and targets
    # calculate values along first axis
    predictions = np.argmax(activation2.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions==y)
    
    # Log tracking info
    losses.append(loss)
    accuracies.append(accuracy)
    iterations.append(iteration)

    # Print accuracy
    print('acc:', accuracy)

    # If loss is smaller - print and save weights and biases aside
    if loss < lowest_loss:
        print('New set of weights found, iteration:', iteration, 
            'loss:', loss, 'acc:', accuracy)
        best_dense1_weights = dense_1.weights.copy()
        best_dense1_biases = dense_1.biases.copy()
        best_dense2_weights = dense_2.weights.copy()
        best_dense2_biases = dense_2.biases.copy()
        lowest_loss = loss

    # Revert weights and losses
    else:
        dense_1.weights = best_dense1_weights.copy()
        dense_1.biases = best_dense1_biases.copy()
        dense_2.weights = best_dense2_weights.copy()
        dense_2.biases = best_dense2_biases.copy()

# Plot 1: Spiral input
plt.figure(figsize=(6, 5))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg', s=20)
plt.title(f"Input: Spiral Dataset\nLoss: {loss:.3f} | Accuracy: {accuracy:.2%}")
plt.xlabel("Feature X[0]")
plt.ylabel("Feature X[1]")
plt.grid(True)
plt.tight_layout()
plt.savefig("spiral_dataset.png")

# Plot 2: ReLU activations (Layer 1)
plt.figure(figsize=(6, 4))
for i in range(3):
    plt.plot(activation1.output[:, i], label=f"Neuron {i+1}")
plt.title("Layer 1 Activations (ReLU)")
plt.xlabel("Sample Index")
plt.ylabel("Activation Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("relu_activations.png")

# Plot 3: Softmax outputs
plt.figure(figsize=(6, 4))
for i in range(3):
    plt.plot(activation2.output[:, i], label=f"Class {i}")
plt.title("Softmax Output Probabilities")
plt.xlabel("Sample Index")
plt.ylabel("Probability")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("softmax_outputs.png")

# Plot 4: Loss over iterations
plt.figure(figsize=(6, 4))
plt.plot(iterations, losses, label='Loss')
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Loss over Iterations")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("loss_curve.png")

# Plot 5: Accuracy over iterations
plt.figure(figsize=(6, 4))
plt.plot(iterations, accuracies, label='Accuracy', color='green')
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.title("Accuracy over Iterations")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("accuracy_curve.png")

plt.show()

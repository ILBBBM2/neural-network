import os
import logging
import keyboard
import time
import pandas as pd
import cv2
import numpy as np
import pickle
from typing import Tuple, List, Optional

#logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
restoring = False
class ActivationFunctions:
    @staticmethod
    def relu(x):

        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x):

        return (x > 0).astype(np.float32)
    
    @staticmethod
    def softmax(x):

        x_clipped = np.clip(x, -100, 100)
        exp_x = np.exp(x_clipped - np.max(x_clipped, axis=1, keepdims=True))

        return exp_x / (np.sum(exp_x, axis=1, keepdims=True) + 1e-10)
    
    @staticmethod
    def tanh(x):

        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x):

        return 1 - np.tanh(x) ** 2

class NeuralNetwork:
    def __init__(self, input_size: int, hidden_size: int, output_size: int, learning_rate: float = 0.01, 
                 use_batch_norm: bool = True, dropout_rate: float = 0.2, decay_rate: float = 0.95):

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        self.decay_rate = decay_rate

        input_scale = np.sqrt(2.0 / (input_size + hidden_size))
        hidden_scale = np.sqrt(2.0 / (hidden_size + output_size))
        
        self.weights_input_hidden = np.random.randn(input_size, hidden_size).astype(np.float32) * input_scale
        self.weights_hidden_output = np.random.randn(hidden_size, output_size).astype(np.float32) * hidden_scale
        #inint with pos to prevent dead brain cells
        self.bias_hidden = np.random.randn(1, hidden_size).astype(np.float32) * 0.01
        self.bias_output = np.random.randn(1, output_size).astype(np.float32) * 0.01

        if use_batch_norm:
            self.gamma_hidden = np.ones((1, hidden_size), dtype=np.float32)
            self.beta_hidden = np.zeros((1, hidden_size), dtype=np.float32)
            self.gamma_output = np.ones((1, output_size), dtype=np.float32)
            self.beta_output = np.zeros((1, output_size), dtype=np.float32)

        self.activations = ActivationFunctions()

        self.momentum = 0.9
        self.v_weights_input_hidden = np.zeros_like(self.weights_input_hidden)
        self.v_weights_hidden_output = np.zeros_like(self.weights_hidden_output)
        self.v_bias_hidden = np.zeros_like(self.bias_hidden)
        self.v_bias_output = np.zeros_like(self.bias_output)
        
        logging.info(f" nn initialized with input_size={input_size}, hidden_size={hidden_size}, output_size={output_size}")
    
    def batch_normalize(self, x, gamma, beta, training: bool = True):

        if not self.use_batch_norm:
            return x
        mean = np.mean(x, axis=0, keepdims=True)
        var = np.var(x, axis=0, keepdims=True) + 1e-5                
        x_norm = (x - mean) / np.sqrt(var)       
        return gamma * x_norm + beta    
    def dropout(self, x, training: bool = True):
        if not training or self.dropout_rate == 0:
            return x
        mask = np.random.binomial(1, 1 - self.dropout_rate, size=x.shape) / (1 - self.dropout_rate)
        mask = np.nan_to_num(mask, nan=0.0, posinf=1.0, neginf=0.0)
        return x * mask
    def forward(self, X, training: bool = True):   
        if np.any(np.isnan(X)):
            X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
        self.hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        if self.use_batch_norm:
            self.hidden_layer_input = self.batch_normalize(self.hidden_layer_input, self.gamma_hidden, self.beta_hidden, training)
        self.hidden_layer_output = self.activations.relu(self.hidden_layer_input)
        
        self.hidden_layer_output = self.dropout(self.hidden_layer_output, training)

        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        if self.use_batch_norm:
            self.output_layer_input = self.batch_normalize(self.output_layer_input, self.gamma_output, self.beta_output, training)
        self.output = self.activations.softmax(self.output_layer_input)    
        if np.any(np.isnan(self.output)):
            self.output = np.nan_to_num(self.output, nan=0.0, posinf=1.0, neginf=0.0)
        return self.hidden_layer_output, self.output_layer_input, self.output
    def backward(self, X, y):
        #no nan input
        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
            y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0)              
        output_error = self.output - y              
        hidden_error = np.dot(output_error, self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.activations.relu_derivative(self.hidden_layer_output)
        
        
        if self.dropout_rate > 0:
            hidden_delta = hidden_delta * (self.hidden_layer_output > 0)
        
        
        d_weights_hidden_output = np.dot(self.hidden_layer_output.T, output_error)
        d_bias_output = np.sum(output_error, axis=0, keepdims=True)
        
        d_weights_input_hidden = np.dot(X.T, hidden_delta)
        d_bias_hidden = np.sum(hidden_delta, axis=0, keepdims=True)
        
        
        d_weights_hidden_output = np.clip(d_weights_hidden_output, -5, 5)
        d_bias_output = np.clip(d_bias_output, -5, 5)
        d_weights_input_hidden = np.clip(d_weights_input_hidden, -5, 5)
        d_bias_hidden = np.clip(d_bias_hidden, -5, 5)
        
        #nn ommentum (change in future)
        self.v_weights_hidden_output = self.momentum * self.v_weights_hidden_output - self.learning_rate * d_weights_hidden_output
        self.v_bias_output = self.momentum * self.v_bias_output - self.learning_rate * d_bias_output
        
        self.v_weights_input_hidden = self.momentum * self.v_weights_input_hidden - self.learning_rate * d_weights_input_hidden
        self.v_bias_hidden = self.momentum * self.v_bias_hidden - self.learning_rate * d_bias_hidden
        
        
        self.v_weights_hidden_output = np.clip(self.v_weights_hidden_output, -5, 5)
        self.v_bias_output = np.clip(self.v_bias_output, -5, 5)
        self.v_weights_input_hidden = np.clip(self.v_weights_input_hidden, -5, 5)
        self.v_bias_hidden = np.clip(self.v_bias_hidden, -5, 5)
        
        
        self.weights_hidden_output += self.v_weights_hidden_output
        self.bias_output += self.v_bias_output
        
        self.weights_input_hidden += self.v_weights_input_hidden
        self.bias_hidden += self.v_bias_hidden
        
        #update batch normalization - IMPROVED
        if self.use_batch_norm:
            
            gamma_update = self.learning_rate * np.sum(output_error * self.output_layer_input, axis=0, keepdims=True)
            beta_update = self.learning_rate * np.sum(output_error, axis=0, keepdims=True)
            
            
            gamma_update = np.clip(gamma_update, -0.5, 0.5)
            beta_update = np.clip(beta_update, -0.5, 0.5)
            
            self.gamma_output -= gamma_update
            self.beta_output -= beta_update
            
            
            gamma_hidden_update = self.learning_rate * np.sum(hidden_delta * self.hidden_layer_input, axis=0, keepdims=True)
            beta_hidden_update = self.learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)
            
            
            gamma_hidden_update = np.clip(gamma_hidden_update, -0.5, 0.5)
            beta_hidden_update = np.clip(beta_hidden_update, -0.5, 0.5)
            
            self.gamma_hidden -= gamma_hidden_update
            self.beta_hidden -= beta_hidden_update
    
    def train(self, X, y, epochs: int, batch_size: int = 32, patience: int = 3):
        n_samples = X.shape[0]
        losses = []
        accuracies = []
        
        
        best_loss = float('inf')
        best_weights = None
        best_bias_hidden = None
        best_bias_output = None
        best_gamma_hidden = None
        best_beta_hidden = None
        best_gamma_output = None
        best_beta_output = None
        patience_counter = 0
    
        print("\n press 'k' to stop running and make predictions")
    
        for epoch in range(epochs):
            
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
    
            epoch_loss = 0
            correct_predictions = 0
            
            
            if epoch > 0:
                self.learning_rate = self.initial_learning_rate * (self.decay_rate ** epoch)
                if epoch % 5 == 0:
                    print(f"learning rate: {self.learning_rate:.6f}")
    
            
            for i in range(0, n_samples, batch_size):
                
                if keyboard.is_pressed('k'):
                    print("\n training stoppped early....")
                    return losses, accuracies  
    
                batch_X = X_shuffled[i:i + batch_size]
                batch_y = y_shuffled[i:i + batch_size]
    
                
                _, _, output = self.forward(batch_X, training=True)
    
                
                batch_loss = -np.mean(np.sum(batch_y * np.log(output + 1e-10), axis=1))
                
                
                if np.isnan(batch_loss):
                    print(f"WARNING NAN VALUES DETECTED AT EPOCH {epoch+1}, batch {i//batch_size+1}. Skipping batch......")
                    continue
                
                epoch_loss += batch_loss * len(batch_X)
                
                predictions = np.argmax(output, axis=1)
                true_labels = np.argmax(batch_y, axis=1)
                correct_predictions += np.sum(predictions == true_labels)
    
                
                self.backward(batch_X, batch_y)
    
            
            epoch_loss /= n_samples
            epoch_accuracy = correct_predictions / n_samples
            
            #NAN SUCKS AHHAHAHAHAHHAHAHAHAHH
            if np.isnan(epoch_loss):
                print(f"WARNING: NAN VALUES DETECTED AT EPOCH{epoch+1}. using previous loss value....")
                if losses:
                    epoch_loss = losses[-1]
                else:
                    epoch_loss = 1.0  #default avlue
            
            losses.append(float(epoch_loss))
            accuracies.append(float(epoch_accuracy))
            
            
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
                
                #best weight save
                best_weights = {
                    'weights_input_hidden': self.weights_input_hidden.copy(),
                    'weights_hidden_output': self.weights_hidden_output.copy(),
                    'bias_hidden': self.bias_hidden.copy(),
                    'bias_output': self.bias_output.copy(),
                    'gamma_hidden': self.gamma_hidden.copy() if self.use_batch_norm else None,
                    'beta_hidden': self.beta_hidden.copy() if self.use_batch_norm else None,
                    'gamma_output': self.gamma_output.copy() if self.use_batch_norm else None,
                    'beta_output': self.beta_output.copy() if self.use_batch_norm else None
                }
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n early stopping at {epoch+1}. loss has been increasing for{patience} epochs.")
                    print(f"restoring best weight from {epoch-patience+1} with loss {best_loss:.6f}")
                    
                    #restoraation of best results because nn tends to go off the rails after a while
                    self.weights_input_hidden = best_weights['weights_input_hidden']
                    self.weights_hidden_output = best_weights['weights_hidden_output']
                    self.bias_hidden = best_weights['bias_hidden']
                    self.bias_output = best_weights['bias_output']
                    if self.use_batch_norm:
                        self.gamma_hidden = best_weights['gamma_hidden']
                        self.beta_hidden = best_weights['beta_hidden']
                        self.gamma_output = best_weights['gamma_output']
                        self.beta_output = best_weights['beta_output']
                    restoring = True
                    break
    
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"epoch {epoch + 1}, loss: {float(epoch_loss):.6f}, accuracy: {float(epoch_accuracy):.4f}")
            else:
                print(f"epoch {epoch + 1}, loss: {float(epoch_loss):.6f}, accuracy: {float(epoch_accuracy):.4f}", end='\r')
        #cout << "\n"; equivilant because python sucks
        print() 
        return losses, accuracies
    
    def predict(self, X):
        _, _, output = self.forward(X, training=False)
        return np.argmax(output, axis=1)
    
    def evaluate(self, X, y):
        _, _, output = self.forward(X, training=False)
        predictions = np.argmax(output, axis=1)
        true_labels = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == true_labels)
        return accuracy
    #saving the actual nn to a file lmao
    def save(self, filepath: str) -> None:
        
        state = {
            #vars that need to be saved (edit in future)
            'weights_input_hidden': self.weights_input_hidden,
            'weights_hidden_output': self.weights_hidden_output,
            'bias_hidden': self.bias_hidden,
            'bias_output': self.bias_output,
            'gamma_hidden': self.gamma_hidden if self.use_batch_norm else None,
            'beta_hidden': self.beta_hidden if self.use_batch_norm else None,
            'gamma_output': self.gamma_output if self.use_batch_norm else None,
            'beta_output': self.beta_output if self.use_batch_norm else None,
            'v_weights_input_hidden': self.v_weights_input_hidden,
            'v_weights_hidden_output': self.v_weights_hidden_output,
            'v_bias_hidden': self.v_bias_hidden,
            'v_bias_output': self.v_bias_output,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'initial_learning_rate': self.initial_learning_rate,
            'learning_rate': self.learning_rate,
            'use_batch_norm': self.use_batch_norm,
            'dropout_rate': self.dropout_rate,
            'momentum': self.momentum,
            'decay_rate': self.decay_rate
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        logging.info(f"nn saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'NeuralNetwork':
        #load nn from file
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        nn = cls(
            input_size=state['input_size'],
            hidden_size=state['hidden_size'],
            output_size=state['output_size'],
            learning_rate=state['learning_rate'],
            use_batch_norm=state['use_batch_norm'],
            dropout_rate=state['dropout_rate'],
            decay_rate=state.get('decay_rate', 0.95)  #default to 0.95 if not found
        )
        nn.weights_input_hidden = state['weights_input_hidden']
        nn.weights_hidden_output = state['weights_hidden_output']
        nn.bias_hidden = state['bias_hidden']
        nn.bias_output = state['bias_output']
        nn.gamma_hidden = state['gamma_hidden']
        nn.beta_hidden = state['beta_hidden']
        nn.gamma_output = state['gamma_output']
        nn.beta_output = state['beta_output']
        nn.v_weights_input_hidden = state['v_weights_input_hidden']
        nn.v_weights_hidden_output = state['v_weights_hidden_output']
        nn.v_bias_hidden = state['v_bias_hidden']
        nn.v_bias_output = state['v_bias_output']
        nn.momentum = state['momentum']
        nn.initial_learning_rate = state.get('initial_learning_rate', state['learning_rate'])
        logging.info(f"Neural network loaded from {filepath}")
        return nn

class DataLoader:
    def __init__(self, metadata_file: str, dataset_dir: str, image_size: Tuple[int, int] = (300, 300)):

        self.metadata_file = metadata_file
        self.dataset_dir = dataset_dir
        self.image_size = image_size
        
    def load_and_preprocess_data(self):
        #load MD
        data = pd.read_csv(self.metadata_file)
        
        features = []
        labels = []
        class_names = sorted(data['class'].unique().tolist())
        
        for _, row in data.iterrows():
            image_name = row['image']
            label = row['class']
            
            
            image_path = os.path.join(self.dataset_dir, label, image_name).replace("\\", "/")
            
            if not os.path.exists(image_path):
                logging.warning(f"file not found: {image_path}")
                continue
            
            #processing
            try:
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    logging.warning(f"could not load image: {image_path}")
                    continue
                
                
                image = cv2.resize(image, self.image_size)
                
                
                image = image / 255.0
                
                
                features.append(image.flatten())
                
                
                label_idx = class_names.index(label)
                one_hot = np.zeros(len(class_names))
                one_hot[label_idx] = 1
                labels.append(one_hot)
                
            except Exception as e:
                logging.error(f"error processing image {image_path}: {str(e)}")
                continue
        
        # Convert to NumPy arrays
        features = np.array(features, dtype=np.float32)
        labels = np.array(labels, dtype=np.float32)
        
        return features, labels, class_names

def main():
    
    METADATA_FILE = "metadata.csv"
    DATASET_DIR = "datasets/Brain Tumor Data Set"
    IMAGE_SIZE = (300, 300)
    INPUT_SIZE = IMAGE_SIZE[0] * IMAGE_SIZE[1]
    HIDDEN_SIZE = 512  
    OUTPUT_SIZE = 2  
    EPOCHS = 50
    BATCH_SIZE = 16  
    LEARNING_RATE = 0.0001  
    DECAY_RATE = 0.99  
    DROPOUT_RATE = 0.2  
    MODEL_SAVE_PATH = "neural_network_state.pkl"
    
    #load dtaa
    data_loader = DataLoader(METADATA_FILE, DATASET_DIR, IMAGE_SIZE)
    X, y, class_names = data_loader.load_and_preprocess_data()
    
    print(f"\nloaded {len(X)} samples with {len(class_names)} classes: {class_names}")
    
    
    nn = NeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, LEARNING_RATE, 
                      use_batch_norm=True, dropout_rate=DROPOUT_RATE, decay_rate=DECAY_RATE)
    
    #model pkl file check
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"found the  saved model at {MODEL_SAVE_PATH}. Loading...")
        nn = NeuralNetwork.load(MODEL_SAVE_PATH)
    
    
    print(f"Starting training with {EPOCHS} epochs, batch size {BATCH_SIZE}, learning rate {LEARNING_RATE}")
    losses, accuracies = nn.train(X, y, EPOCHS, BATCH_SIZE, patience=5)
    
    #save
    nn.save(MODEL_SAVE_PATH)
    
    final_accuracy = nn.evaluate(X, y)
    print(f"\nfinal model accuracy: {final_accuracy:.4f}")
    nn.save(MODEL_SAVE_PATH)
    
    test_image_path = os.path.join(DATASET_DIR, "tumor", "Cancer (1).jpg")
    if restoring:
        nn.save(MODEL_SAVE_PATH)
    if os.path.exists(test_image_path):
        
        test_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
        test_image = cv2.resize(test_image, IMAGE_SIZE)
        test_image = test_image / 255.0
        test_image = test_image.flatten().reshape(1, -1)
        
        
        prediction = nn.predict(test_image)
        predicted_class = class_names[int(prediction[0])]
        
        print(f"\ntest image prediction: {predicted_class}")

if __name__ == "__main__":
    main() 
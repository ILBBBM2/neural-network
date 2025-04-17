import cv2
import numpy as np
import pickle
import logging
import os
from typing import Tuple


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ActivationFunctions:
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def softmax(x):
        x_clipped = np.clip(x, -100, 100)
        exp_x = np.exp(x_clipped - np.max(x_clipped, axis=1, keepdims=True))
        return exp_x / (np.sum(exp_x, axis=1, keepdims=True) + 1e-10)
class NeuralNetwork:
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activations = ActivationFunctions()
    def forward(self, X):
        hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        if hasattr(self, 'gamma_hidden'):
            hidden_layer_input = self.batch_normalize(hidden_layer_input, self.gamma_hidden, self.beta_hidden)
        hidden_layer_output = self.activations.relu(hidden_layer_input)
        output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output) + self.bias_output
        if hasattr(self, 'gamma_output'):
            output_layer_input = self.batch_normalize(output_layer_input, self.gamma_output, self.beta_output)
        output = self.activations.softmax(output_layer_input)
        
        return output

    def batch_normalize(self, x, gamma, beta):
        mean = np.mean(x, axis=0, keepdims=True)
        var = np.var(x, axis=0, keepdims=True) + 1e-5
        x_norm = (x - mean) / np.sqrt(var)
        return gamma * x_norm + beta

    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)

    @classmethod
    def load(cls, filepath: str) -> 'NeuralNetwork':
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        nn = cls(
            input_size=state['input_size'],
            hidden_size=state['hidden_size'],
            output_size=state['output_size']
        )
        nn.weights_input_hidden = state['weights_input_hidden']
        nn.weights_hidden_output = state['weights_hidden_output']
        nn.bias_hidden = state['bias_hidden']
        nn.bias_output = state['bias_output']
        
        if state.get('use_batch_norm', False):
            nn.gamma_hidden = state['gamma_hidden']
            nn.beta_hidden = state['beta_hidden']
            nn.gamma_output = state['gamma_output']
            nn.beta_output = state['beta_output']
        
        logging.info(f"Neural network loaded from {filepath}")
        return nn

def preprocess_image(image_path: str, image_size: Tuple[int, int] = (300, 300)) -> np.ndarray:
    
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"couldnt load image: {image_path}")
    
    image = cv2.resize(image, image_size)
    image = image / 255.0  
    return image.flatten().reshape(1, -1)

def main():
    
    MODEL_PATH = "neural_network_state.pkl"
    IMAGE_SIZE = (300, 300)
    CLASS_NAMES = ["no_tumor", "tumor"]  
    DATASET_DIR = "input"
    
    try:
        nn = NeuralNetwork.load(MODEL_PATH)
        logging.info("nn model loaded successfully")
    except Exception as e:
        logging.error(f"failed to load model: {str(e)}")
        return

    
    test_image_path = os.path.join(DATASET_DIR, "test.jpg")
    try:
        test_image = preprocess_image(test_image_path, IMAGE_SIZE)
        
        
        prediction = nn.predict(test_image)
        predicted_class = CLASS_NAMES[int(prediction[0])]
        
        print(f"\nprediction for image: {predicted_class}")
        print("please visit then nearest doctor for medical assistance!")
        
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
    
    
    print("\nPress Enter to exit................................")
    input()

if __name__ == "__main__":
    main() 
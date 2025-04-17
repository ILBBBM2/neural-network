# **a neural network for medical image classification**

> [!NOTE]
> this is a pre-release build, expect some bugs and compatibility issues

```
$$\                        $$\                                   $$\                  $$\    $$\      $$\         $$\$$\ 
$$ |                       \__|                                  $$ |                 $$ |   $$ |     \__|        $$ $$ |
$$$$$$$\  $$$$$$\  $$$$$$\ $$\       $$$$$$\$$$$\  $$$$$$\  $$$$$$$ |$$$$$$\        $$$$$$\  $$$$$$$\ $$\ $$$$$$$\$$ $$ |
$$  __$$\ \____$$\$$  __$$\$$ |      $$  _$$  _$$\ \____$$\$$  __$$ $$  __$$\       \_$$  _| $$  __$$\$$ $$  _____$$ $$ |
$$ |  $$ |$$$$$$$ $$ |  \__$$ |      $$ / $$ / $$ |$$$$$$$ $$ /  $$ $$$$$$$$ |        $$ |   $$ |  $$ $$ \$$$$$$\ \__\__|
$$ |  $$ $$  __$$ $$ |     $$ |      $$ | $$ | $$ $$  __$$ $$ |  $$ $$   ____|        $$ |$$\$$ |  $$ $$ |\____$$\       
$$ |  $$ \$$$$$$$ $$ |     $$ |      $$ | $$ | $$ \$$$$$$$ \$$$$$$$ \$$$$$$$\         \$$$$  $$ |  $$ $$ $$$$$$$  $$\$$\ 
\__|  \__|\_______\__|     \__|      \__| \__| \__|\_______|\_______|\_______|         \____/\__|  \__\__\_______/\__\__|
                                                                                                                         
                                                                                                                         
                                                                                                                         
```

written in python with numpy. no external dependencies beyond standard libraries. OpenCV used for image resizing and such

requirements for training:
- python 3.6+
- numpy
- opencv
- pickle (for model serialization)

## **the neural network architecture:**

this project implements a custom neural network for medical image classification, specifically designed to detect brain tumors in mri scans. the architecture consists of:

1. **input layer**: processes 300x300 grayscale images (90,000 input neurons)
2. **hidden layer**: 512 neurons with relu activation
3. **output layer**: 2 neurons with softmax activation (binary classification)

the network includes several advanced features:
- batch normalization for improved training stability
- dropout (20%) to prevent overfitting
- momentum-based gradient descent
- learning rate decay (0.95 per epoch)


a simple visualisation:
![Capture](https://private-user-images.githubusercontent.com/166126131/434835303-79c0e30f-9b1c-4308-8679-5587848ba572.PNG?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDQ4OTgxODYsIm5iZiI6MTc0NDg5Nzg4NiwicGF0aCI6Ii8xNjYxMjYxMzEvNDM0ODM1MzAzLTc5YzBlMzBmLTliMWMtNDMwOC04Njc5LTU1ODc4NDhiYTU3Mi5QTkc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUwNDE3JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MDQxN1QxMzUxMjZaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT01Yjk1YzE3YzVhMzBkMzY0MjkzN2E4ZTg3ZWQxMjg5NTNiNzVlMDYxZjFlMWMyMWQ5MmNiNDMxZjljNjM5NDVhJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.EupK2zrxH0yOf9TUXQqlTQXirXThnlbU1F2zdWXRL5I)


## **the idea:**

medical image classification is a critical application of machine learning, but many existing solutions require significant computational resources and large datasets. this project was born from the challenge of creating an efficient neural network that could be trained on limited medical imaging data.

the neural network is designed to be lightweight while maintaining good classification performance. it uses a simple architecture with 512 neurons, but incorporates modern techniques like batch normalization and dropout to improve generalization, with a 92% success rate in identifying whether the patient has cancer or not

## **challenges of neural network implementation:**

implementing a neural network from scratch without using established frameworks like tensorflow or pytorch presented several challenges:

1. **numerical stability**: the softmax activation function can lead to numerical overflow with large inputs. this was addressed by clipping values and using a numerically stable implementation.

2. **gradient explosion/vanishing**: without proper initialization, the network suffered from gradient issues. this was solved by scaling the initial weights based on the layer sizes.

3. **training instability**: early training was unstable with oscillations in loss. implementing momentum and learning rate decay helped stabilize the training process.

4. **memory constraints**: processing large images requires significant memory. the solution was to normalize pixel values to [0,1] range and use float32 instead of float64.

5. **early stopping**: the network tended to overfit after extended training. a patience-based early stopping mechanism was implemented to save the best model weights.

## **implementation details:**

the neural network is implemented in a modular way with separate classes for:

1. **activationfunctions**: contains static methods for relu and softmax activations
2. **neuralnetwork**: the main network class with forward/backward passes
3. **dataloader**: handles image loading and preprocessing

key implementation features:

```python
def forward(self, x):
    hidden_layer_input = np.dot(x, self.weights_input_hidden) + self.bias_hidden
    if hasattr(self, 'gamma_hidden'):
        hidden_layer_input = self.batch_normalize(hidden_layer_input, self.gamma_hidden, self.beta_hidden)
    hidden_layer_output = self.activations.relu(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output) + self.bias_output
    if hasattr(self, 'gamma_output'):
        output_layer_input = self.batch_normalize(output_layer_input, self.gamma_output, self.beta_output)
    output = self.activations.softmax(output_layer_input)
    
    return output
```

the batch normalization implementation helps stabilize training:

```python
def batch_normalize(self, x, gamma, beta):
    mean = np.mean(x, axis=0, keepdims=True)
    var = np.var(x, axis=0, keepdims=True) + 1e-5
    x_norm = (x - mean) / np.sqrt(var)
    return gamma * x_norm + beta
```

## **model persistence and prediction:**

the trained model can be saved to a pkl file and loaded later for predictions:

```python
@classmethod
def load(cls, filepath: str) -> 'neuralnetwork':
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
    
    if state.get('use_batch_norm', false):
        nn.gamma_hidden = state['gamma_hidden']
        nn.beta_hidden = state['beta_hidden']
        nn.gamma_output = state['gamma_output']
        nn.beta_output = state['beta_output']
    
    return nn
```

## **performance optimization:**

several techniques were used to optimize the neural network's performance:

1. **vectorized operations**: using numpy's vectorized operations instead of loops
2. **efficient memory usage**: minimizing memory allocations during training
3. **early stopping**: saving the best model weights to prevent overfitting
4. **batch processing**: processing data in batches for better memory efficiency

## **future improvements:**

- implement convolutional layers for better feature extraction
- add support for multi-class classification
- develop a web interface for easy model deployment
- implement transfer learning from pre-trained models
- add confidence scores to predictions

## **usage using .py**

### requirements
- python 3.6+
- numpy
- opencv
- pickle (for model serialization)

1. download the brain tumor dataset used from [here](https://www.kaggle.com/datasets/preetviradiya/brian-tumor-dataset)
2. copy and paste the dataset into
   ```
   datasets/
   ```
3. rename the two subfolders of the dataset to normal and tumor respectively
4. place the image you want to test into the
   ```
   datasets/
   ```
   directory and rename it as Cancer (1).jpg
5. alternatively you can edit line 475
  ```python
  test_image_path = os.path.join(DATASET_DIR, "tumor", "{replace name here}.jpg")
  ``` 
   
6. run the file:
   ```
   headtracker.py
   ```
7. train the model until satisfied!

the prediction script will load the trained model and classify the specified image as either "tumor" or "no_tumor".

## **usage using the downloaded exe**

1. place the image you want to test in the directory
   ```
   input/[place image here]
   ```
2. rename your image as "test.jpg"
3. run the executable

# Developing a Neural Network Regression Model
# Date: 11-05-2026
## AIM
To develop a neural network regression model for the given dataset.

## THEORY
The objective of this experiment is to design, implement, and evaluate a Deep Learning–based Neural Network regression model to predict a continuous output variable from a given set of input features.

In many real-world applications—such as house price prediction, temperature forecasting, sales estimation, or demand prediction—the relationship between input variables and the output is non-linear and complex. Traditional statistical models often fail to capture these patterns effectively. Deep Learning models, particularly Artificial Neural Networks (ANNs), are capable of learning such complex relationships through multiple hidden layers and non-linear activation functions.

In this experiment, a dataset containing multiple independent variables (features) and a dependent variable (target) is provided. The task is to preprocess the data, construct a neural network regression architecture, train the model using backpropagation and gradient descent, and evaluate its performance using appropriate regression metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and R² score.

The experiment aims to understand how network architecture, learning rate, number of epochs, and activation functions affect the accuracy of regression predictions and to demonstrate the effectiveness of deep learning in solving regression problems.

## Neural Network Model
<img width="743" height="623" alt="image" src="https://github.com/user-attachments/assets/c91c53e2-abfc-4adf-a2b0-072e26f9ecaa" />


## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: Harisudhan S

### Register Number:212224240048

```python

class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(1, 8)
        self.fc2=nn.Linear(8, 10)
        self.fc3=nn.Linear(10, 1)
        self.relu=nn.ReLU()
        self.history={'loss':[]}
  def forward(self, x):
        x=self.relu(self.fc1(x))
        x=self.relu(self.fc2(x))
        x=self.fc3(x)
        return x


# Initialize the Model, Loss Function, and Optimizer

lig=NeuralNet()
criterion=nn.MSELoss()
optimizer=optim.RMSprop(lig. parameters(), lr=0.001);



def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range (epochs):
        optimizer. zero_grad()
        loss=criterion(ai_brain(X_train), y_train)
        loss. backward()
        optimizer.step()
        lig .history['loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')

```

### Dataset Information

<img width="210" height="248" alt="image" src="https://github.com/user-attachments/assets/85b09959-7b95-46c1-aebd-0cfa8adedb9f" />

### OUTPUT
<img width="166" height="415" alt="image" src="https://github.com/user-attachments/assets/2d1d0354-86e5-4ce8-8184-2571ab83b027" />


### Training Loss Vs Iteration Plot
<img width="818" height="564" alt="image" src="https://github.com/user-attachments/assets/5d3926da-9955-44bc-a9e1-d44cc7d024f7" />


### New Sample Data Prediction
<img width="334" height="51" alt="image" src="https://github.com/user-attachments/assets/7ded04f9-9181-4779-9900-1ad507ffe255" />


## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.

# Developing a Neural Network Regression Model
# Date: 11-05-2026
## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Explain the problem statement

## Neural Network Model
Include the neural network model diagram.

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

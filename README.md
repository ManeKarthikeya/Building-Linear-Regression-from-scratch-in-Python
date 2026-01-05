# 📈 Building Linear Regression from Scratch in Python

## 🧮 Complete Implementation of Linear Regression Algorithm with Gradient Descent

A **from-scratch implementation** of the Linear Regression algorithm using pure Python and NumPy. This educational project demonstrates the fundamental mathematics and optimization techniques behind one of the most widely used machine learning algorithms, complete with gradient descent optimization and practical salary prediction application.

## 🎯 Project Overview

This project implements Linear Regression **without relying on machine learning libraries** like scikit-learn. It provides a deep understanding of:
- The mathematical foundations of Linear Regression
- Gradient Descent optimization algorithm
- Weight and bias parameter updates
- Model training and prediction mechanics

## 📚 Mathematical Foundations

### 🧠 Linear Regression Equation
```
Y = wX + b
```
Where:
- **Y**: Dependent Variable (Target)
- **X**: Independent Variable (Feature)
- **w**: Weight parameter
- **b**: Bias parameter

### ⚙️ Gradient Descent Algorithm
The optimization algorithm used to minimize the loss function by iteratively updating parameters:

**Parameter Update Rules:**
```
w = w - α * dw
b = b - α * db
```

**Gradients Calculation:**
```
ŷ⁽ⁱ⁾ = wx⁽ⁱ⁾ + b
dw = (1/m) * Σ (ŷ⁽ⁱ⁾ - y⁽ⁱ⁾) * x⁽ⁱ⁾
db = (1/m) * Σ (ŷ⁽ⁱ⁾ - y⁽ⁱ⁾)
```

### 📊 Key Concepts
- **Learning Rate (α)**: Step size at each iteration while moving toward minimum loss
- **Gradient**: Partial derivatives indicating direction of steepest ascent
- **Loss Function**: Mean Squared Error (MSE) minimized during training
- **Iterations**: Number of times gradient descent updates parameters

## 🛠️ Technical Implementation

### 🏗️ Custom Linear Regression Class

#### Class Structure:
```python
class Linear_Regression():
    def __init__(self, learning_rate, no_of_iterations):
        # Initialize hyperparameters
    
    def fit(self, X, Y):
        # Training function with gradient descent
    
    def update_weights(self):
        # Gradient descent weight updates
    
    def predict(self, X):
        # Make predictions using learned parameters
```

#### Core Methods:
1. **`__init__`**: Initialize learning rate and iteration count
2. **`fit`**: Train model using gradient descent
3. **`update_weights`**: Compute gradients and update parameters
4. **`predict`**: Generate predictions using learned weights

### 🔄 Gradient Descent Implementation
```python
def update_weights(self):
    # Calculate predictions
    Y_prediction = self.predict(self.X)
    
    # Compute gradients
    dw = -(2 * (self.X.T).dot(self.Y - Y_prediction)) / self.m
    db = -(2 * np.sum(self.Y - Y_prediction)) / self.m
    
    # Update parameters
    self.w = self.w - self.learning_rate * dw
    self.b = self.b - self.learning_rate * db
```

## 📊 Dataset & Application

### 💰 Salary Prediction Problem
**Dataset**: Salary vs. Years of Experience
- **Features**: Years of work experience
- **Target**: Salary amount
- **Samples**: Multiple employee records

### 🧪 Model Training Results
```
Final Equation: salary = 9514 * (experience) + 23697
```

**Learned Parameters:**
- **Weight (w)**: 9514.40
- **Bias (b)**: 23697.08

## 🚀 Quick Start

### Prerequisites
```bash
python >= 3.8
pip install numpy pandas matplotlib scikit-learn
```

### Installation & Usage

1. **Clone the repository**:
```bash
git clone https://github.com/ManeKarthikeya/Building-Linear-Regression-from-Scratch.git
cd Building-Linear-Regression-from-Scratch
```

2. **Run the salary prediction system**:
```bash
python building_linear_regression_from_scratch_in_python.py
```

3. **Input years of experience** when prompted:
```python
# Example interaction:
# Please enter the years of experience: 5.5
# Years of Experience entered: 5.5
# Predicted Salary for 5.5 years of experience: 76026.28
```

### Manual Usage Example
```python
# Import custom Linear Regression class
from linear_regression import Linear_Regression

# Create and train model
model = Linear_Regression(learning_rate=0.02, no_of_iterations=1000)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
```

## 📁 Project Structure

```
Linear-Regression-from-Scratch/
├── building_linear_regression_from_scratch_in_python.py  # Main implementation
├── linear_regression.py                                 # Custom Linear Regression class(if needed separate from Main implementation)
├── salary_data.csv                                     # Salary dataset
└── README.md                                          # Project documentation
```

## 🧪 Testing & Validation

### 🎯 Model Evaluation
- **Training-Validation Split**: 67-33 split using scikit-learn
- **Visual Validation**: Scatter plot with regression line
- **Mathematical Verification**: Manual calculation checks
- **Hyperparameter Tuning**: Learning rate and iteration optimization

### 📈 Visualization
```python
plt.scatter(X_test, Y_test, color='red')          # Actual values
plt.plot(X_test, test_data_prediction, color='blue')  # Predicted line
plt.xlabel('Work Experience')
plt.ylabel('Salary')
plt.title('Salary vs Experience')
plt.show()
```

## 🎯 Educational Value

### 🎓 Learning Objectives
1. **Understanding Linear Regression Mathematics**
   - Slope-intercept form interpretation
   - Gradient calculation derivation
   - Loss function minimization

2. **Implementing Gradient Descent**
   - Parameter update mechanics
   - Learning rate impact
   - Convergence criteria

3. **Custom Algorithm Development**
   - Object-oriented programming for ML
   - Numerical computation with NumPy
   - Model evaluation techniques

### 🔍 Key Insights
- **Weight Interpretation**: Each year of experience adds ~$9,514 to salary
- **Bias Interpretation**: Base salary for zero experience is ~$23,697
- **Learning Rate Sensitivity**: Too high causes divergence, too slow causes slow convergence
- **Iteration Trade-off**: More iterations improve accuracy but increase computation time

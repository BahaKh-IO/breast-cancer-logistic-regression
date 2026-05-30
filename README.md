🩺 Breast Cancer Wisconsin Classification using Logistic Regression

A machine learning project that implements Logistic Regression from scratch using NumPy to classify breast tumors as Malignant (M) or Benign (B) using the Breast Cancer Wisconsin Dataset.

Unlike library-based implementations, this project builds the entire learning algorithm manually, including:

Sigmoid activation
Cross-Entropy Loss
L2 Regularization
Gradient Computation
Gradient Descent Optimization
Prediction Function
📊 Dataset

The project uses the Breast Cancer Wisconsin Diagnostic Dataset, which contains numerical features extracted from digitized images of breast mass cell nuclei.

Target Classes
Label	Meaning
0	Benign
1	Malignant
Features

The dataset contains 30 numerical features such as:

Radius
Texture
Perimeter
Area
Smoothness
Compactness
Concavity
Symmetry
Fractal Dimension
🧠 Logistic Regression Model

Logistic Regression estimates the probability that a sample belongs to the positive class:

P(y=1∣x)=σ(z)

where

z=w
T
x+b

and

σ(z)=
1+e
−z
1
	​

⚙️ Functions and Mathematical Formulas
1. Sigmoid Function
def sigmoid(z):
    return (1/(1+np.exp(-z)))
Formula
σ(z)=
1+e
−z
1
	​

Purpose

Transforms any real-valued number into a probability between 0 and 1.

2. Cost Function (Cross-Entropy + L2 Regularization)
def cost(x, y, w, b, lambda_):
Cross-Entropy Loss
J(w,b)=−
m
1
	​

i=1
∑
m
	​

[y
(i)
log(
y
^
	​

(i)
)+(1−y
(i)
)log(1−
y
^
	​

(i)
)]

where

y
^
	​

=σ(w
T
x+b)
L2 Regularization
2m
λ
	​

j=1
∑
n
	​

w
j
2
	​

Total Cost
J(w,b)=−
m
1
	​

i=1
∑
m
	​

[y
(i)
log(
y
^
	​

(i)
)+(1−y
(i)
)log(1−
y
^
	​

(i)
)]+
2m
λ
	​

j=1
∑
n
	​

w
j
2
	​

Purpose
Measures prediction error.
Penalizes large weights to reduce overfitting.
3. Gradient Computation
def gradient(x,y,w,b,lambda_):
Weight Gradient
∂w
j
	​

∂J
	​

=
m
1
	​

i=1
∑
m
	​

(
y
^
	​

(i)
−y
(i)
)x
j
(i)
	​

+
m
λ
	​

w
j
	​

Bias Gradient
∂b
∂J
	​

=
m
1
	​

i=1
∑
m
	​

(
y
^
	​

(i)
−y
(i)
)
Purpose

Computes how much each parameter contributes to the prediction error.

4. Gradient Descent
def gradient_descent(x,y,w,b,alpha,lambda_,iterations):
Weight Update
w
j
	​

:=w
j
	​

−α
∂w
j
	​

∂J
	​

Bias Update
b:=b−α
∂b
∂J
	​


where

α = learning rate
Purpose

Iteratively updates parameters to minimize the cost function.

5. Prediction Function
def predict(x,w,b):
Probability Computation
y
^
	​

=σ(w
T
x+b)
Decision Rule
Prediction={
1
0
	​

if 
y
^
	​

≥0.5
otherwise
	​

Purpose

Converts probabilities into class labels.

🔄 Data Preprocessing

Feature scaling is applied before training:

x_scaled=(x-np.mean(x))/np.std(x)
Standardization Formula
x
scaled
	​

=
σ
x−μ
	​


where:

μ = mean
σ = standard deviation
Why?

Standardization helps Gradient Descent converge faster and improves numerical stability.

🚀 Training Configuration
Parameter	Value
Learning Rate	0.05
Regularization λ	1
Iterations	15000
Initialization	Zeros
📈 Training Process

During training:

Compute predictions using the sigmoid function.
Calculate cost.
Compute gradients.
Update parameters using gradient descent.
Store cost history for visualization.

The loss curve is plotted to monitor convergence.

plt.plot(costs)
🎯 Evaluation

Model performance is measured using classification accuracy:

Accuracy=
Total Predictions
Correct Predictions
	​

×100
np.mean(predictions == labels) * 100

Metrics reported:

Training Accuracy
Test Accuracy
🛠️ Technologies Used
Python
NumPy
Pandas
Matplotlib
📂 Project Structure
├── Breast Cancer Wisconsin - Logistic Regression.ipynb
├── data.csv
├── README.md
📚 Learning Objectives

This project demonstrates:

Logistic Regression from scratch
Binary Classification
Cross-Entropy Loss
L2 Regularization
Gradient Descent Optimization
Feature Scaling
Model Evaluation
📄 License

This project is released under the MIT License.

Author

Developed as an educational implementation of Logistic Regression to understand the mathematics and optimization process behind binary classification models.

# 🩺 Breast Cancer Wisconsin Classification using Logistic Regression

A machine learning project that implements Logistic Regression from scratch using NumPy to classify breast tumors as Malignant (M) or Benign (B) using the Breast Cancer Wisconsin Dataset.

## Dataset

The Breast Cancer Wisconsin Diagnostic Dataset contains 30 numerical features extracted from breast mass images.

Target labels:

| Label | Class |
|---------|---------|
| 0 | Benign |
| 1 | Malignant |

---
## Features
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

## Mathematical Background

### 1. Sigmoid Function

```python
def sigmoid(z):
    return 1/(1+np.exp(-z))
```

Formula:

σ(z) = 1 / (1 + e^(-z))

Maps any real number to the interval (0,1).

---

### 2. Logistic Regression Hypothesis

```text
z = wᵀx + b
ŷ = σ(z)
```

where:

- x = feature vector
- w = weight vector
- b = bias
- ŷ = predicted probability

---

### 3. Cost Function

Cross-Entropy Loss:

```text
J = -(1/m) Σ [y log(ŷ) + (1-y) log(1-ŷ)]
```

L2 Regularization:

```text
(λ/2m) Σ w²
```

Total Cost:

```text
J = CrossEntropy + Regularization
```

---

### 4. Gradient Computation

Weight Gradient:

```text
∂J/∂w = (1/m) Xᵀ(ŷ - y) + (λ/m)w
```

Bias Gradient:

```text
∂J/∂b = (1/m) Σ(ŷ - y)
```

---

### 5. Gradient Descent Update

Weights:

```text
w := w - α(∂J/∂w)
```

Bias:

```text
b := b - α(∂J/∂b)
```

where α is the learning rate.

---

### 6. Prediction Rule

```text
ŷ = σ(wᵀx + b)
```

Decision:

```text
If ŷ ≥ 0.5 → Class 1
Else → Class 0
```

---

## Data Preprocessing

Standardization:

```text
x_scaled = (x - μ) / σ
```

where:

- μ = mean
- σ = standard deviation

---

## Training Parameters

| Parameter | Value |
|------------|---------|
| Learning Rate | 0.05 |
| Lambda | 1 |
| Iterations | 2400 |

---

## Results

After training the Logistic Regression model using Gradient Descent with L2 Regularization:

| Metric | Score |
|---------|---------|
| Training Accuracy | 90.89% |
| Test Accuracy | 94.12% |

### Interpretation

- The model successfully generalizes to unseen data, achieving **94.12% accuracy on the test set**.
- The small difference between training and testing accuracy suggests that the model is not significantly overfitting.
- L2 Regularization helps improve generalization by preventing excessively large weight values.

```
Train Accuracy = 90.89%
Test Accuracy  = 94.12%
```

---

## Technologies

- Python
- NumPy
- Pandas
- Matplotlib

---

## Project Structure

```text
├── Breast Cancer Wisconsin - Logistic Regression.ipynb
├── data.csv
├── README.md
```

---

## License

This project is released under the MIT License.

## Author

Developed as an educational implementation of Logistic Regression to understand the mathematics and optimization process behind binary classification models.

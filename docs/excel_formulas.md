# Mathematical Processes as Excel Formulas

This document provides Excel/Spreadsheet formula equivalents for the various mathematical and machine learning operations used throughout the project.

## 1. Data Preprocessing

### Standard Scaler (Z-Score Normalization)
Standardizes features by removing the mean and scaling to unit variance.
* **Math:** `z = (x - u) / s`
* **Excel Reference:** Assuming data starts in `A2`.
    * Mean: `=AVERAGE(A:A)`
    * Standard Deviation: `=STDEV.P(A:A)`
    * **Formula (in B2):** `=(A2 - AVERAGE(A$2:A$100)) / STDEV.P(A$2:A$100)`

### Min-Max Scaler
Scales features to a given range, usually between 0 and 1.
* **Math:** `x_scaled = (x - min(x)) / (max(x) - min(x))`
* **Excel Reference:**
    * **Formula (in B2):** `=(A2 - MIN(A$2:A$100)) / (MAX(A$2:A$100) - MIN(A$2:A$100))`

---

## 2. Convolutional Neural Networks (CNN & EfficientNet)

### Convolution (2D)
A dot product between a filter (kernel) and a receptive field of the input.
* **Example:** 3x3 input matrix in `A1:C3`, 3x3 kernel in `E1:G3`.
* **Formula:** `=SUMPRODUCT(A1:C3, E1:G3)`

### ReLU Activation Function
Returns the input if positive, otherwise zero.
* **Math:** `f(x) = max(0, x)`
* **Excel Reference:** Assuming value is in `A1`.
    * **Formula:** `=MAX(0, A1)`

### Sigmoid Activation Function
Squashes the input value between 0 and 1.
* **Math:** `f(x) = 1 / (1 + e^-x)`
* **Excel Reference:** Assuming value is in `A1`.
    * **Formula:** `=1 / (1 + EXP(-A1))`

### Max Pooling (2D)
Takes the maximum value within a window.
* **Example:** 2x2 pooling window in `A1:B2`.
* **Formula:** `=MAX(A1:B2)`

### Average Pooling (2D)
Takes the average value within a window.
* **Example:** 2x2 pooling window in `A1:B2`.
* **Formula:** `=AVERAGE(A1:B2)`

### Dense / Fully Connected Layer (Linear Transformation)
Calculates weights dot inputs plus bias.
* **Math:** `y = W*x + b`
* **Example:** Inputs in `A1:A5`, Weights in `B1:B5`, Bias in `C1`.
* **Formula:** `=SUMPRODUCT(A1:A5, B1:B5) + C1`

---

## 3. Traditional Machine Learning Models

### K-Nearest Neighbors (KNN) - Euclidean Distance
Measures the straight-line distance between two points.
* **Math:** `d(p, q) = sqrt(sum((p_i - q_i)^2))`
* **Example:** Point 1 in `A2:C2`, Point 2 in `A3:C3`.
* **Formula:** `=SQRT(SUMXMY2(A2:C2, A3:C3))`

### Support Vector Machine (SVM) - Linear Kernel Distance
Calculates the decision function score.
* **Math:** `f(x) = w • x + b`
* **Example:** Support Vector / Weights in `A2:C2`, Input features in `X2:Z2`, Bias bias in `D2`.
* **Formula:** `=SUMPRODUCT(A2:C2, X2:Z2) + D2`

### Multiple Linear Regression
Predicts the output value as a linear combination of inputs.
* **Math:** `y = B0 + B1*x1 + B2*x2 + ... + Bn*xn`
* **Excel Reference:** If inputs `x` are in `A2:C2` and coefficients `B1..Bn` are in `X2:Z2`, with Intercept `B0` in `W2`.
* **Formula:** `=SUMPRODUCT(A2:C2, X2:Z2) + W2`
* **Using Built-in Function:** `=TREND(known_y's, known_x's, new_x's)` or `=LINEST()`

### Decision Tree / Random Forest (Split Condition)
A node decision based on a threshold.
* **Math:** `if feature <= threshold, path 1, else path 2`
* **Example:** Feature value in `A2`, Threshold in `B2`.
* **Formula:** `=IF(A2 <= B2, "Left Child", "Right Child")`

---

## 4. Evaluation Metrics

### Mean Absolute Error (MAE)
* **Math:** `(1/n) * sum(|y - y_pred|)`
* **Example:** True values in `A2:A10`, Predicted values in `B2:B10`.
* **Formula:** `=SUMPRODUCT(ABS(A2:A10 - B2:B10)) / COUNT(A2:A10)`

### Mean Squared Error (MSE)
* **Math:** `(1/n) * sum((y - y_pred)^2)`
* **Example:** True values in `A2:A10`, Predicted values in `B2:B10`.
* **Formula:** `=SUMXMY2(A2:A10, B2:B10) / COUNT(A2:A10)`

### Root Mean Squared Error (RMSE)
* **Math:** `sqrt(MSE)`
* **Formula:** `=SQRT(SUMXMY2(A2:A10, B2:B10) / COUNT(A2:A10))`

### R-Squared (Coefficient of Determination)
* **Formula:** `=RSQ(A2:A10, B2:B10)` (Note: Excel RSQ expects `known_y's, known_x's` or in this context `True Values, Predicted Values`)

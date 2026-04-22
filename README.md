# LSTM Model for Sarcasm Detection

## Key Steps:

### 1. Data Preparation
-   Loading and converting `datasets` objects into pandas DataFrames.
-   Tokenizing text comments using `Tokenizer`.
-   Padding sequences to a uniform length for LSTM input.

### 2. Model Building (Initial & Refined)
-   **Baseline model**: A simple Baseline model
-   **LSTM model**: A simple LSTM model was constructed with an `Embedding` layer, an `LSTM` layer, and a `Dense` output layer.
-   **L2 Regularization**: L2 regularization was added to the `Bidirectional(LSTM)` layer to combat overfitting.

### 3. Model Training
-   The model was trained on the prepared training data (`X_train_pad`, `y_train`) and validated on `X_val_pad`, `y_val`.

### 4. Model Evaluation
-   The trained model was evaluated on the test set (`X_test_pad`, `y_test`) to assess its generalization performance.
-   Training and validation accuracy/loss curves were plotted to diagnose overfitting.

## Initial Challenges and Solutions:
-   **Severe Overfitting**: Initially, the model showed high training accuracy but very low test accuracy (around 17%), indicating severe overfitting.
    -   **Solution 1**: Added `kernel_regularizer=regularizers.l2(0.001)` to the `Bidirectional(LSTM)` layer. This significantly reduced overfitting and led to a substantial improvement in test accuracy.

## Current Performance:
View corresponding txt files for performance review and eval

## Model files
- baseline.py
- lstm.py
- bidir_lstm.py

\
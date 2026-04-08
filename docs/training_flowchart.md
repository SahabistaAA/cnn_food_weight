# Neural Network Training Flowchart for Excel Manualization

This document provides a visual flowchart of the mathematical training process for a neural network. This process is designed to be replicated step-by-step in an Excel spreadsheet to manually trace how the model learns.

## 1. Training Process Flowchart

```mermaid
graph TD
    %% Initialization Phase
    subgraph Initiation
        A[Start] --> B("Load Dataset: Input X (4x4), True Label Y (Weight)")
        B --> C("Initialize Kernels K1, K2 (3x3)<br>and Biases b1, b2")
        C --> D(Set Learning Rate: e.g., 0.01)
    end

    %% Forward Pass
    subgraph "Forward Pass (Predictions)"
        D --> F["<b>1. Convolution:</b><br>4x4 Input * 3x3 Kernel<br>(Result: 2x2 Feature Map)"]
        F --> G["<b>2. Activation (ReLU):</b><br>f(x) = MAX(0, x)<br>(Remove Negative Values)"]
        G --> G2["<b>3. Flattening:</b><br>Convert 2x2 matrix into<br>a single column of 4 values"]
        G2 --> G3["<b>4. Dense Layer:</b><br>prediction = (Features * Weights) + Bias"]
    end
    
    %% Error Calculation
    subgraph "Error Calculation"
        G3 --> H["<b>5. Calculate Loss (MSE):</b><br>Error = (True Y - Pred Y)²"]
    end

    %% Backward Pass
    subgraph "Backward Pass & Update"
        H --> I["<b>6. Backpropagation:</b><br>Calculate how much K and b<br>contributed to the Error"]
        I --> J["<b>7. Weight Update:</b><br>New_K = Old_K - (LR * Gradient)"]
    end

    J --> K[End of Iteration]
    K --> D

    %% Completion Phase
    K -- "After many iterations" --> N(End Training)
    N --> O([Model Optimized for Weight Prediction])

    %% Styling
    classDef init fill:#d4ebf2,stroke:#333,stroke-width:2px;
    classDef forward fill:#d5f5e3,stroke:#333,stroke-width:2px;
    classDef back fill:#fadbd8,stroke:#333,stroke-width:2px;
    classDef loss fill:#fcf3cf,stroke:#333,stroke-width:2px;
    
    class A,B,C,D init;
    class F,G,G2,G3 forward;
    class H loss;
    class I,J back;
```

## 2. Step-by-Step for your Excel Sheet

### Step 1: Initialization
*   **Input ($X$):** $4 \times 4$ matrix. Normalize values between 0.0 and 1.0 (Pixel / 255).
*   **Kernel ($K$):** $3 \times 3$ matrix with small random numbers ($ -0.2$ to $0.2$).
*   **Bias ($b$):** Start with 0.

### Step 2: Convolution (Forward Pass)
*   **Formula:** `=SUMPRODUCT(Input_Slice, $Kernel) + $Bias`
*   Create a **$2 \times 2$** output area for each kernel ($K_1, K_2$).

### Step 3: Activation (ReLU)
*   **Formula:** `=MAX(0, Conv_Output_Cell)`
*   This makes the model "Non-Linear" and is essential for learning.

### Step 4: Flattening & Dense Layer
*   Copy your $2 \times 2$ results into a single column (4 rows).
*   Multiply these 4 numbers by 4 "Dense Weights" using `=SUMPRODUCT()` to get your final predicted weight.

### Step 5: Loss Calculation
*   **Formula:** `=(Actual_Weight - Predicted_Weight)^2`
*   Your goal is to make this number as small as possible in the next step!

### Step 6: Backpropagation (The "Training")
*   This is where you adjust your $K$ and $b$ values based on the Error. In Excel, you can use **"Solver"** or manual derivatives to see how changing a kernel value drops the Loss.

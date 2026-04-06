# Neural Network from Scratch in Go 

A simple neural network built **from scratch in Go (Golang)** using matrix operations from Gonum — no ML frameworks.

## Features

- Binary classification (initial)
- Multiclass classification (Iris dataset)
- Softmax + Cross-Entropy loss
- Backpropagation implemented manually
- Train/Test split with accuracy evaluation

---

## Dataset

Uses the Iris dataset:

- Setosa
- Versicolor
- Virginica

---

## How it works

1. Forward pass  
2. Softmax output  
3. Cross-entropy loss  
4. Backpropagation  
5. Gradient descent  

---

## Results

- Train Accuracy: ~95–100%
- Test Accuracy: ~85–95%

---

## Run

```bash
go run .
## Neural Network from Scratch in Go

Built a neural network completely from scratch in Go (Golang) 
without using any machine learning libraries 
to understand how modern frameworks like PyTorch actually work under the hood.

## Why this project?

Most ML projects use high-level libraries like PyTorch or TensorFlow.

This project answers:

What is actually happening inside loss.backward()?

## Features
-Neural network implemented from first principles
-Multiclass classification (Iris dataset)
-Softmax + Cross-Entropy loss
-Backpropagation implemented manually
-Train/Test split with accuracy evaluation
-CLI support for experimentation
## Architecture

Input Layer (4 features)--> Hidden Layer (8 neurons, sigmoid) --> Output Layer (3 neurons, softmax)

## Results
-Train Accuracy: ~95–100%
-Test Accuracy:  ~90–97%
## Interpretation
-Setosa is linearly separable → near-perfect predictions
-Versicolor vs Virginica overlap → occasional confusion
-Model generalizes well despite small dataset
## From Scratch vs Frameworks (Conceptual Comparison)
|Concept |	This Project (Go)	|PyTorch (Typical)|
| :--- | :----: | ----: |
| Data representation |	mat.Dense	|torch.Tensor|
|Layers	|Manual matrices|	nn.Linear
|Activation|	Custom sigmoid|	Built-in
|Softmax	|Manually implemented	|Inside loss
|Loss	|Manual cross-entropy	|CrossEntropyLoss
|Backpropagation|Explicit gradients|	loss.backward()
|Optimization	|Manual updates	|optimizer.step()
## Key Insight

High-level ML frameworks don’t remove complexity — they hide it.

By building everything manually, it becomes clear that:

loss.backward()  = gradient computation
optimizer.step() = weight updates
CrossEntropyLoss = softmax + log + loss combined
## What I Learned
-How gradients flow through a neural network
-Why activation functions affect learning
-How loss functions guide optimization
-Why data preprocessing (normalization, shuffling) is critical
-The difference between learning ML vs using ML libraries
## Project Structure
-neural-net-go/
-├── data/
-├── model/
-├── utils/
-├── main.go
-├── iris.csv
-├── README.md
## Run
go run . -epochs=3000 -lr=0.01 -hidden=8
## Future Improvements
-Mini-batch training
-Model persistence (save/load weights)
-Visualization of decision boundaries
-Deeper architectures
-Port to PyTorch for direct comparison
## Contributing

Feel free to fork and experiment!
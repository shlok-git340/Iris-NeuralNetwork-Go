package utils

import "math"

func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func SigmoidDerivative(x float64) float64 {
	return x * (1 - x)
}
func Softmax(x []float64) []float64 {
	var sum float64
	out := make([]float64, len(x))

	for i, v := range x {
		out[i] = math.Exp(v)
		sum += out[i]
	}

	for i := range out {
		out[i] /= sum
	}

	return out
}

// hidden = sigmoid(X * wHidden + bHidden)
// output = sigmoid(hidden * wOut + bOut)

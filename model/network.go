package model

import (
	"math/rand"

	"github.com/shlok-git340/neural-net/utils"
	"gonum.org/v1/gonum/mat"
)

type NeuralNet struct {
	WHidden *mat.Dense
	WOut    *mat.Dense
	BHidden *mat.Dense
	BOut    *mat.Dense
}

func randomMatrix(r, c int) *mat.Dense {
	data := make([]float64, r*c)

	for i := range data {
		data[i] = rand.Float64()*2 - 1 // range: [-1, 1]
	}

	return mat.NewDense(r, c, data)
}

// Input = 2 features
// Hidden = 2 neurons
// Output = 1 neuron

// Matrix	Shape
// wHidden	2×2
// WOut		2×1
// BHidden	1×2
// BOut		1×1
func NewNetwork(input, hidden, output int) *NeuralNet {
	return &NeuralNet{
		WHidden: randomMatrix(input, hidden),
		WOut:    randomMatrix(hidden, output),
		BHidden: randomMatrix(1, hidden),
		BOut:    randomMatrix(1, output),
	}
}
func (nn *NeuralNet) Forward(x *mat.Dense) *mat.Dense {
	var hidden mat.Dense
	hidden.Mul(x, nn.WHidden)
	hidden.Add(&hidden, nn.BHidden)
	hidden.Apply(func(i int, j int, v float64) float64 {
		return utils.Sigmoid(v)
	}, &hidden)
	var output mat.Dense
	output.Mul(&hidden, nn.WOut)
	output.Add(&output, nn.BOut)

	rows, cols := output.Dims()
	for i := 0; i < rows; i++ {

		row := make([]float64, cols)
		for j := 0; j < cols; j++ {
			row[j] = output.At(i, j)
		}

		row = utils.Softmax(row)

		for j := 0; j < cols; j++ {
			output.Set(i, j, row[j])
		}
	}
	return &output
}
func (nn *NeuralNet) Train(x, y *mat.Dense, lr float64) {
	var hidden mat.Dense
	hidden.Mul(x, nn.WHidden)
	hidden.Add(&hidden, nn.BHidden)

	hidden.Apply(func(i, j int, v float64) float64 {
		return utils.Sigmoid(v)
	}, &hidden)

	var output mat.Dense
	output.Mul(&hidden, nn.WOut)
	output.Add(&output, nn.BOut)

	output.Apply(func(i, j int, v float64) float64 {
		return utils.Sigmoid(v)
	}, &output)

	var outputError mat.Dense
	outputError.Sub(&output, y)

	var outputGradient mat.Dense
	outputGradient.Sub(&output, y)

	hiddenT := mat.DenseCopyOf(hidden.T())
	var deltaWOut mat.Dense
	deltaWOut.Mul(hiddenT, &outputGradient)
	deltaWOut.Scale(lr, &deltaWOut)
	nn.WOut.Sub(nn.WOut, &deltaWOut)

	var deltaBOut mat.Dense
	deltaBOut.Scale(lr, &outputGradient)
	nn.BOut.Sub(nn.BOut, &deltaBOut)

	WOutT := mat.DenseCopyOf(nn.WOut.T())
	var hiddenError mat.Dense
	hiddenError.Mul(&outputGradient, WOutT)

	var hiddenGradient mat.Dense
	hiddenGradient.Apply(func(i, j int, v float64) float64 {
		return utils.SigmoidDerivative(v)
	}, &hidden)
	hiddenGradient.MulElem(&hiddenGradient, &hiddenError)

	xT := mat.DenseCopyOf(x.T())
	var deltaWHidden mat.Dense
	deltaWHidden.Mul(xT, &hiddenGradient)
	deltaWHidden.Scale(lr, &deltaWHidden)
	nn.WHidden.Sub(nn.WHidden, &deltaWHidden)

	var deltaBHidden mat.Dense
	deltaBHidden.Scale(lr, &hiddenGradient)
	nn.BHidden.Sub(nn.BHidden, &deltaBHidden)
}

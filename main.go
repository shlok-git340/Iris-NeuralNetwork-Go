package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/shlok-git340/neural-net/data"
	"github.com/shlok-git340/neural-net/model"
	"gonum.org/v1/gonum/mat"
)

func main() {
	rand.Seed(time.Now().UnixNano())
	inputs, targets := data.LoadIrisCSV("data/iris.csv")
	normalize(inputs)
	total := len(inputs)
	rand.Shuffle(total, func(i, j int) {
		inputs[i], inputs[j] = inputs[j], inputs[i]
		targets[i], targets[j] = targets[j], targets[i]
	})

	split := int(0.8 * float64(total))

	trainX := inputs[:split]
	trainY := targets[:split]

	testX := inputs[split:]
	testY := targets[split:]
	nn := model.NewNetwork(4, 8, 3)
	var totalLoss float64

	for epoch := 0; epoch < 5000; epoch++ {
		totalLoss = 0

		for i := range trainX {
			x := mat.NewDense(1, 4, trainX[i])
			y := mat.NewDense(1, 3, trainY[i])

			out := nn.Forward(x)

			for j := 0; j < 3; j++ {
				totalLoss += -y.At(0, j) * math.Log(out.At(0, j)+1e-9)
			}

			nn.Train(x, y, 0.01)
		}

		correct := 0

		for i := range testX {
			x := mat.NewDense(1, 4, testX[i])
			out := nn.Forward(x)

			outRow := []float64{
				out.At(0, 0),
				out.At(0, 1),
				out.At(0, 2),
			}

			pred := argmax(outRow)
			actual := argmax(testY[i])

			if pred == actual {
				correct++
			}
		}

		accuracy := float64(correct) / float64(len(testX))
		fmt.Println("\nTest Accuracy:", accuracy)
		correctTrain := 0

		for i := range trainX {
			x := mat.NewDense(1, 4, trainX[i])
			out := nn.Forward(x)

			outRow := []float64{
				out.At(0, 0),
				out.At(0, 1),
				out.At(0, 2),
			}

			pred := argmax(outRow)
			actual := argmax(trainY[i])

			if pred == actual {
				correctTrain++
			}
		}

		trainAcc := float64(correctTrain) / float64(len(trainX))
		fmt.Println("Train Accuracy:", trainAcc)

		if epoch%500 == 0 {
			fmt.Println("Epoch:", epoch, "Loss:", totalLoss)
		}

	}
	fmt.Println("\nPredictions:")

	for i := range inputs {
		x := mat.NewDense(1, 4, inputs[i])
		out := nn.Forward(x)

		outRow := []float64{
			out.At(0, 0),
			out.At(0, 1),
			out.At(0, 2),
		}
		pred := argmax(outRow)

		fmt.Printf("Input: %v → %v → Predicted class: %d\n",
			inputs[i],
			outRow,
			pred,
		)
	}
}
func normalize(data [][]float64) {
	for i := range data {
		for j := range data[i] {
			data[i][j] /= 10.0
		}
	}
}
func argmax(row []float64) int {
	maxIdx := 0
	for i := 1; i < len(row); i++ {
		if row[i] > row[maxIdx] {
			maxIdx = i
		}
	}
	return maxIdx
}

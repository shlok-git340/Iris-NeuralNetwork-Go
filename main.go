package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/shlok-git340/neural-net/data"
	"github.com/shlok-git340/neural-net/model"
	"gonum.org/v1/gonum/mat"
)

func main() {
	epochs := flag.Int("epochs", 3000, "Number of training epochs")
	learningRate := flag.Float64("lr", 0.01, "Learning Rate")
	hiddenSize := flag.Int("hidden", 8, "Number of hidden neurons")
	dataPath := flag.String("data", "data/iris.csv", "Path to dataset")
	version := flag.Bool("version", false, "Show version")
	flag.Usage = func() {
		fmt.Println("Usage: iris-nn [options]")
		fmt.Println("\nOptions:")
		flag.PrintDefaults()
	}
	flag.Parse()
	if *version {
		fmt.Println("neural-net v1.0.0")
		return
	}
	fmt.Println("===== Iris Neural Network CLI =====")
	fmt.Printf("Epochs: %d\n", *epochs)
	fmt.Printf("Learning Rate: %.4f\n", *learningRate)
	fmt.Printf("Hidden Neurons: %d\n\n", *hiddenSize)
	fmt.Printf("Dataset: %s\n\n", *dataPath)
	fmt.Println("starting neural network")
	rand.Seed(time.Now().UnixNano())
	inputs, targets := data.LoadIrisCSV(*dataPath)
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
	nn := model.NewNetwork(4, *hiddenSize, 3)

	for epoch := 0; epoch < *epochs; epoch++ {
		var totalLoss float64

		for i := range trainX {
			x := mat.NewDense(1, 4, trainX[i])
			y := mat.NewDense(1, 3, trainY[i])

			out := nn.Forward(x)

			for j := 0; j < 3; j++ {
				totalLoss += -y.At(0, j) * math.Log(out.At(0, j)+1e-9)
			}

			nn.Train(x, y, *learningRate)
		}
		if epoch%500 == 0 {
			fmt.Printf("Epoch: %d Loss: %.4f\n", epoch, totalLoss)
		}
	}
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

	correctTest := 0

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
			correctTest++
		}
	}

	testAcc := float64(correctTest) / float64(len(testX))

	fmt.Println("\n===== RESULTS =====")
	fmt.Printf("Train Accuracy: %.2f%%\n", trainAcc*100)
	fmt.Printf("Test Accuracy: %.2f%%\n", testAcc*100)

	fmt.Println("\nPredictions:")

	for i := 0; i < 10 && i < len(testX); i++ {
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
func argmax(row []float64) int {
	maxIdx := 0
	for i := 1; i < len(row); i++ {
		if row[i] > row[maxIdx] {
			maxIdx = i
		}
	}
	return maxIdx
}

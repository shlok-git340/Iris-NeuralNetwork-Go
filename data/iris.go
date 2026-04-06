package data

import (
	"encoding/csv"
	"os"
	"strconv"
	"strings"
)

func LoadIrisCSV(path string) ([][]float64, [][]float64) {

	file, err := os.Open(path)
	if err != nil {
		panic(err)
	}
	defer file.Close()
	reader := csv.NewReader(file)
	var inputs [][]float64
	var targets [][]float64

	rowIndex := 0

	for {
		record, err := reader.Read()

		if err != nil {
			break // EOF reached
		}

		// skip header if needed
		if rowIndex == 0 {
			rowIndex++
			continue
		}

		// parse features
		x := make([]float64, 4)
		for i := 0; i < 4; i++ {
			val, err := strconv.ParseFloat(record[i], 64)
			if err != nil {
				panic(err)
			}
			x[i] = val / 10.0
		}

		// parse label
		label := strings.ToLower(strings.TrimSpace(record[4]))

		var y []float64
		switch label {
		case "setosa":
			y = []float64{1, 0, 0}
		case "versicolor":
			y = []float64{0, 1, 0}
		case "virginica":
			y = []float64{0, 0, 1}
		default:
			panic("Unknown label: " + label)
		}

		inputs = append(inputs, x)
		targets = append(targets, y)

		rowIndex++
	}

	return inputs, targets
}

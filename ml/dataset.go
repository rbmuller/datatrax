// Package ml provides machine learning algorithms and utilities
// implemented in pure Go with zero external dependencies.
package ml

import (
	"encoding/csv"
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"strings"
)

// Dataset holds feature matrix X and label vector Y for ML tasks.
type Dataset struct {
	X [][]float64
	Y []float64
}

// NewDataset creates a new Dataset from feature matrix x and label vector y.
func NewDataset(x [][]float64, y []float64) *Dataset {
	return &Dataset{X: x, Y: y}
}

// LoadCSV loads a CSV file, parses all values as float64, and extracts
// the column at targetCol as the label vector Y. The remaining columns
// become the feature matrix X. The first row is treated as a header
// if it cannot be parsed as floats.
func LoadCSV(path string, targetCol int) (*Dataset, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("ml: open csv: %w", err)
	}
	defer f.Close()

	reader := csv.NewReader(f)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("ml: read csv: %w", err)
	}

	if len(records) == 0 {
		return nil, fmt.Errorf("ml: csv file is empty")
	}

	// Detect header row by trying to parse the first row
	startRow := 0
	if len(records) > 1 {
		_, parseErr := strconv.ParseFloat(strings.TrimSpace(records[0][0]), 64)
		if parseErr != nil {
			startRow = 1
		}
	}

	totalCols := len(records[startRow])
	if targetCol < 0 || targetCol >= totalCols {
		return nil, fmt.Errorf("ml: targetCol %d out of range [0, %d)", targetCol, totalCols)
	}

	rows := len(records) - startRow
	featureCols := totalCols - 1

	x := make([][]float64, rows)
	y := make([]float64, rows)

	for i := startRow; i < len(records); i++ {
		row := records[i]
		if len(row) != totalCols {
			return nil, fmt.Errorf("ml: row %d has %d columns, expected %d", i, len(row), totalCols)
		}

		idx := i - startRow
		features := make([]float64, 0, featureCols)

		for j := 0; j < totalCols; j++ {
			val, err := strconv.ParseFloat(strings.TrimSpace(row[j]), 64)
			if err != nil {
				return nil, fmt.Errorf("ml: parse error at row %d col %d: %w", i, j, err)
			}
			if j == targetCol {
				y[idx] = val
			} else {
				features = append(features, val)
			}
		}
		x[idx] = features
	}

	return &Dataset{X: x, Y: y}, nil
}

// Split divides the dataset into training and test sets based on trainRatio.
// The split is random. trainRatio should be between 0 and 1.
func (d *Dataset) Split(trainRatio float64) (xTrain, xTest [][]float64, yTrain, yTest []float64) {
	n := len(d.X)
	indices := rand.Perm(n)
	splitIdx := int(float64(n) * trainRatio)

	xTrain = make([][]float64, splitIdx)
	yTrain = make([]float64, splitIdx)
	xTest = make([][]float64, n-splitIdx)
	yTest = make([]float64, n-splitIdx)

	for i, idx := range indices {
		if i < splitIdx {
			xTrain[i] = d.X[idx]
			yTrain[i] = d.Y[idx]
		} else {
			xTest[i-splitIdx] = d.X[idx]
			yTest[i-splitIdx] = d.Y[idx]
		}
	}
	return
}

// Shuffle randomly reorders the rows of the dataset in place.
func (d *Dataset) Shuffle() {
	n := len(d.X)
	for i := n - 1; i > 0; i-- {
		j := rand.Intn(i + 1)
		d.X[i], d.X[j] = d.X[j], d.X[i]
		d.Y[i], d.Y[j] = d.Y[j], d.Y[i]
	}
}

// Shape returns the number of rows and feature columns in the dataset.
func (d *Dataset) Shape() (rows, cols int) {
	rows = len(d.X)
	if rows > 0 {
		cols = len(d.X[0])
	}
	return
}

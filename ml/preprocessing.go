package ml

import "math"

// MinMaxScale scales each feature column to the range [0, 1].
// Returns a new matrix; the original is not modified.
func MinMaxScale(data [][]float64) [][]float64 {
	if len(data) == 0 {
		return nil
	}
	rows := len(data)
	cols := len(data[0])
	result := make([][]float64, rows)
	for i := range result {
		result[i] = make([]float64, cols)
	}

	for j := 0; j < cols; j++ {
		minVal := MinOfColumn(data, j)
		maxVal := MaxOfColumn(data, j)
		rangeVal := maxVal - minVal

		for i := 0; i < rows; i++ {
			if rangeVal == 0 {
				result[i][j] = 0
			} else {
				result[i][j] = (data[i][j] - minVal) / rangeVal
			}
		}
	}
	return result
}

// StandardScale applies z-score normalization (mean=0, std=1) to each column.
// Returns a new matrix; the original is not modified.
func StandardScale(data [][]float64) [][]float64 {
	if len(data) == 0 {
		return nil
	}
	rows := len(data)
	cols := len(data[0])
	result := make([][]float64, rows)
	for i := range result {
		result[i] = make([]float64, cols)
	}

	for j := 0; j < cols; j++ {
		mean := MeanOfColumn(data, j)
		std := StdDevOfColumn(data, j)

		for i := 0; i < rows; i++ {
			if std == 0 {
				result[i][j] = 0
			} else {
				result[i][j] = (data[i][j] - mean) / std
			}
		}
	}
	return result
}

// MeanOfColumn computes the arithmetic mean of the specified column.
func MeanOfColumn(data [][]float64, col int) float64 {
	if len(data) == 0 {
		return 0
	}
	sum := 0.0
	for _, row := range data {
		sum += row[col]
	}
	return sum / float64(len(data))
}

// StdDevOfColumn computes the population standard deviation of the specified column.
func StdDevOfColumn(data [][]float64, col int) float64 {
	if len(data) == 0 {
		return 0
	}
	mean := MeanOfColumn(data, col)
	sumSq := 0.0
	for _, row := range data {
		diff := row[col] - mean
		sumSq += diff * diff
	}
	return math.Sqrt(sumSq / float64(len(data)))
}

// MinOfColumn returns the minimum value in the specified column.
func MinOfColumn(data [][]float64, col int) float64 {
	if len(data) == 0 {
		return 0
	}
	minVal := data[0][col]
	for _, row := range data[1:] {
		if row[col] < minVal {
			minVal = row[col]
		}
	}
	return minVal
}

// MaxOfColumn returns the maximum value in the specified column.
func MaxOfColumn(data [][]float64, col int) float64 {
	if len(data) == 0 {
		return 0
	}
	maxVal := data[0][col]
	for _, row := range data[1:] {
		if row[col] > maxVal {
			maxVal = row[col]
		}
	}
	return maxVal
}

package ml

import (
	"math"
	"sort"
)

// KNN implements the K-Nearest Neighbors algorithm for classification.
type KNN struct {
	K        int
	Distance string // "euclidean" or "manhattan"
	xTrain   [][]float64
	yTrain   []float64
}

// KNNConfig holds configuration for creating a KNN classifier.
type KNNConfig struct {
	K        int
	Distance string
}

// NewKNN creates a new KNN classifier with the given configuration.
// Defaults: K=5, Distance="euclidean".
func NewKNN(config KNNConfig) *KNN {
	k := config.K
	if k <= 0 {
		k = 5
	}
	dist := config.Distance
	if dist == "" {
		dist = "euclidean"
	}
	return &KNN{K: k, Distance: dist}
}

// Fit stores the training data for later use during prediction.
func (knn *KNN) Fit(x [][]float64, y []float64) {
	knn.xTrain = x
	knn.yTrain = y
}

// Predict returns predicted class labels for all samples in x by
// majority vote of the K nearest neighbors.
func (knn *KNN) Predict(x [][]float64) []float64 {
	preds := make([]float64, len(x))
	for i, sample := range x {
		preds[i] = knn.predictSingle(sample)
	}
	return preds
}

func (knn *KNN) predictSingle(sample []float64) float64 {
	type neighbor struct {
		dist  float64
		label float64
	}

	neighbors := make([]neighbor, len(knn.xTrain))
	for i, trainSample := range knn.xTrain {
		var d float64
		switch knn.Distance {
		case "manhattan":
			d = manhattanDistance(sample, trainSample)
		default:
			d = euclideanDistance(sample, trainSample)
		}
		neighbors[i] = neighbor{dist: d, label: knn.yTrain[i]}
	}

	sort.Slice(neighbors, func(i, j int) bool {
		return neighbors[i].dist < neighbors[j].dist
	})

	k := knn.K
	if k > len(neighbors) {
		k = len(neighbors)
	}

	// Majority vote
	votes := make(map[float64]int)
	for _, n := range neighbors[:k] {
		votes[n.label]++
	}

	bestLabel := 0.0
	bestCount := 0
	for label, count := range votes {
		if count > bestCount {
			bestCount = count
			bestLabel = label
		}
	}
	return bestLabel
}

// euclideanDistance computes the Euclidean distance between two vectors.
func euclideanDistance(a, b []float64) float64 {
	sum := 0.0
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return math.Sqrt(sum)
}

// manhattanDistance computes the Manhattan distance between two vectors.
func manhattanDistance(a, b []float64) float64 {
	sum := 0.0
	for i := range a {
		sum += math.Abs(a[i] - b[i])
	}
	return sum
}

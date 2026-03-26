package ml

import (
	"math"
	"math/rand"
)

// KMeans implements the K-Means clustering algorithm with K-Means++
// initialization.
type KMeans struct {
	K         int
	MaxIter   int
	Centroids [][]float64
	labels    []int
	inertia   float64
}

// KMeansConfig holds configuration for creating a KMeans clusterer.
type KMeansConfig struct {
	K       int
	MaxIter int
}

// NewKMeans creates a new KMeans clusterer with the given configuration.
// Default MaxIter is 100.
func NewKMeans(config KMeansConfig) *KMeans {
	maxIter := config.MaxIter
	if maxIter <= 0 {
		maxIter = 100
	}
	return &KMeans{
		K:       config.K,
		MaxIter: maxIter,
	}
}

// Fit runs the K-Means algorithm on the data using Lloyd's algorithm
// with K-Means++ initialization.
func (km *KMeans) Fit(x [][]float64) {
	if len(x) == 0 || km.K <= 0 {
		return
	}
	n := len(x)
	dims := len(x[0])

	// K-Means++ initialization
	km.Centroids = make([][]float64, km.K)
	// Pick first centroid randomly
	firstIdx := rand.Intn(n)
	km.Centroids[0] = make([]float64, dims)
	copy(km.Centroids[0], x[firstIdx])

	for c := 1; c < km.K; c++ {
		// Compute squared distances to nearest centroid
		dists := make([]float64, n)
		totalDist := 0.0
		for i := 0; i < n; i++ {
			minDist := math.MaxFloat64
			for j := 0; j < c; j++ {
				d := squaredEuclidean(x[i], km.Centroids[j])
				if d < minDist {
					minDist = d
				}
			}
			dists[i] = minDist
			totalDist += minDist
		}

		// Weighted random selection
		target := rand.Float64() * totalDist
		cumSum := 0.0
		chosen := 0
		for i := 0; i < n; i++ {
			cumSum += dists[i]
			if cumSum >= target {
				chosen = i
				break
			}
		}
		km.Centroids[c] = make([]float64, dims)
		copy(km.Centroids[c], x[chosen])
	}

	km.labels = make([]int, n)

	// Lloyd's algorithm
	for iter := 0; iter < km.MaxIter; iter++ {
		// Assignment step
		changed := false
		for i := 0; i < n; i++ {
			nearest := km.nearestCentroid(x[i])
			if km.labels[i] != nearest {
				km.labels[i] = nearest
				changed = true
			}
		}

		if !changed {
			break
		}

		// Update step
		counts := make([]int, km.K)
		newCentroids := make([][]float64, km.K)
		for c := 0; c < km.K; c++ {
			newCentroids[c] = make([]float64, dims)
		}

		for i := 0; i < n; i++ {
			c := km.labels[i]
			counts[c]++
			for j := 0; j < dims; j++ {
				newCentroids[c][j] += x[i][j]
			}
		}

		for c := 0; c < km.K; c++ {
			if counts[c] > 0 {
				for j := 0; j < dims; j++ {
					newCentroids[c][j] /= float64(counts[c])
				}
			}
		}
		km.Centroids = newCentroids
	}

	// Compute inertia
	km.inertia = 0
	for i := 0; i < n; i++ {
		km.inertia += squaredEuclidean(x[i], km.Centroids[km.labels[i]])
	}
}

// Predict assigns cluster labels to the given data points.
func (km *KMeans) Predict(x [][]float64) []int {
	labels := make([]int, len(x))
	for i, point := range x {
		labels[i] = km.nearestCentroid(point)
	}
	return labels
}

// Inertia returns the sum of squared distances of samples to their
// closest centroid, computed after Fit.
func (km *KMeans) Inertia() float64 {
	return km.inertia
}

func (km *KMeans) nearestCentroid(point []float64) int {
	bestIdx := 0
	bestDist := math.MaxFloat64
	for c, centroid := range km.Centroids {
		d := squaredEuclidean(point, centroid)
		if d < bestDist {
			bestDist = d
			bestIdx = c
		}
	}
	return bestIdx
}

func squaredEuclidean(a, b []float64) float64 {
	sum := 0.0
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return sum
}

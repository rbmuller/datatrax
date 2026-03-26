package main

import (
	"fmt"
	"math/rand"

	"github.com/rbmuller/datatrax/ml"
)

const (
	pointsPerCluster = 50
	numClusters      = 3
)

func main() {
	rand.Seed(42)

	// Generate 3 clearly separated clusters in 2D.
	clusterCenters := [][]float64{
		{0.0, 0.0},
		{10.0, 10.0},
		{20.0, 0.0},
	}

	n := pointsPerCluster * numClusters
	x := make([][]float64, n)
	trueLabels := make([]int, n)

	idx := 0
	for c := 0; c < numClusters; c++ {
		for i := 0; i < pointsPerCluster; i++ {
			x[idx] = []float64{
				clusterCenters[c][0] + rand.NormFloat64()*0.5,
				clusterCenters[c][1] + rand.NormFloat64()*0.5,
			}
			trueLabels[idx] = c
			idx++
		}
	}

	// Fit K-Means
	km := ml.NewKMeans(ml.KMeansConfig{K: numClusters, MaxIter: 100})
	km.Fit(x)
	labels := km.Predict(x)

	fmt.Printf("K-Means Clustering (K=%d)\n", numClusters)
	fmt.Printf("  Total points: %d\n", n)
	fmt.Printf("  Inertia:      %.4f\n", km.Inertia())

	// Print cluster sizes
	clusterSizes := make([]int, numClusters)
	for _, label := range labels {
		clusterSizes[label]++
	}
	fmt.Println("\n  Cluster sizes:")
	for c := 0; c < numClusters; c++ {
		fmt.Printf("    Cluster %d: %d points\n", c, clusterSizes[c])
	}

	// Print centroids
	fmt.Println("\n  Centroids:")
	for c := 0; c < numClusters; c++ {
		fmt.Printf("    Cluster %d: (%.2f, %.2f)\n", c,
			km.Centroids[c][0], km.Centroids[c][1])
	}

	// Show first few assignments
	fmt.Println("\n  Sample assignments (first 5 from each original group):")
	for g := 0; g < numClusters; g++ {
		start := g * pointsPerCluster
		fmt.Printf("    Group %d (center %.0f,%.0f): ",
			g, clusterCenters[g][0], clusterCenters[g][1])
		for i := 0; i < 5; i++ {
			fmt.Printf("cluster_%d ", labels[start+i])
		}
		fmt.Println()
	}
}

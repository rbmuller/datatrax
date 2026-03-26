package main

import (
	"fmt"
	"math/rand"

	"github.com/rbmuller/datatrax/ml"
)

const (
	samplesPerClass = 50
	numClasses      = 3
	numFeatures     = 4
	trainRatio      = 0.8
	kNeighbors      = 5
)

func main() {
	rand.Seed(42)

	// Generate synthetic Iris-like data: 3 classes, 4 features each.
	// Each class has a distinct centroid with Gaussian noise.
	centroids := [][]float64{
		{5.0, 3.4, 1.4, 0.2}, // Setosa-like
		{5.9, 2.8, 4.3, 1.3}, // Versicolor-like
		{6.6, 3.0, 5.6, 2.0}, // Virginica-like
	}

	n := samplesPerClass * numClasses
	x := make([][]float64, n)
	y := make([]float64, n)

	idx := 0
	for class := 0; class < numClasses; class++ {
		for i := 0; i < samplesPerClass; i++ {
			row := make([]float64, numFeatures)
			for f := 0; f < numFeatures; f++ {
				row[f] = centroids[class][f] + rand.NormFloat64()*0.3
			}
			x[idx] = row
			y[idx] = float64(class)
			idx++
		}
	}

	// Split into train/test
	ds := ml.NewDataset(x, y)
	ds.Shuffle()
	xTrain, xTest, yTrain, yTest := ds.Split(trainRatio)

	// Train KNN
	knn := ml.NewKNN(ml.KNNConfig{K: kNeighbors, Distance: "euclidean", Weighted: true})
	knn.Fit(xTrain, yTrain)
	preds := knn.Predict(xTest)

	// Evaluate
	acc := ml.Accuracy(yTest, preds)
	fmt.Printf("KNN Classification (K=%d, weighted)\n", kNeighbors)
	fmt.Printf("  Train samples: %d\n", len(xTrain))
	fmt.Printf("  Test samples:  %d\n", len(xTest))
	fmt.Printf("  Accuracy:      %.2f%%\n", acc*100)

	// Print per-class precision/recall
	classNames := []string{"Setosa", "Versicolor", "Virginica"}
	fmt.Println("\n  Per-class metrics:")
	for c := 0; c < numClasses; c++ {
		p := ml.Precision(yTest, preds, float64(c))
		r := ml.Recall(yTest, preds, float64(c))
		f1 := ml.F1Score(yTest, preds, float64(c))
		fmt.Printf("    %s: Precision=%.2f  Recall=%.2f  F1=%.2f\n",
			classNames[c], p, r, f1)
	}

	// Confusion matrix (simple count per class pair)
	fmt.Println("\n  Confusion Matrix (rows=actual, cols=predicted):")
	cm := make([][]int, numClasses)
	for i := range cm {
		cm[i] = make([]int, numClasses)
	}
	for i := range yTest {
		actual := int(yTest[i])
		predicted := int(preds[i])
		cm[actual][predicted]++
	}
	fmt.Print("           ")
	for c := 0; c < numClasses; c++ {
		fmt.Printf("%-12s", classNames[c])
	}
	fmt.Println()
	for i := 0; i < numClasses; i++ {
		fmt.Printf("    %-7s", classNames[i])
		for j := 0; j < numClasses; j++ {
			fmt.Printf("%-12d", cm[i][j])
		}
		fmt.Println()
	}
}

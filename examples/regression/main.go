package main

import (
	"fmt"
	"math/rand"

	"github.com/rbmuller/datatrax/ml"
)

const (
	numSamples = 100
	slope      = 3.0
	noiseScale = 0.5
	trainRatio = 0.8
)

func main() {
	rand.Seed(42)

	// Generate synthetic data: y = 3x + noise
	x := make([][]float64, numSamples)
	y := make([]float64, numSamples)
	for i := 0; i < numSamples; i++ {
		xi := float64(i) / float64(numSamples) * 10.0
		x[i] = []float64{xi}
		y[i] = slope*xi + rand.NormFloat64()*noiseScale
	}

	// Split into train/test
	ds := ml.NewDataset(x, y)
	ds.Shuffle()
	xTrain, xTest, yTrain, yTest := ds.Split(trainRatio)

	// Train using normal equation (exact solution)
	lr := ml.NewLinearRegression()
	lr.FitNormalEquation(xTrain, yTrain)

	// Predict on test set
	preds := lr.Predict(xTest)

	// Evaluate
	r2 := ml.R2Score(yTest, preds)
	mse := ml.MSE(yTest, preds)
	rmse := ml.RMSE(yTest, preds)
	mae := ml.MAE(yTest, preds)

	fmt.Println("Linear Regression (y = 3x + noise)")
	fmt.Printf("  Train samples: %d\n", len(xTrain))
	fmt.Printf("  Test samples:  %d\n", len(xTest))
	fmt.Printf("  Learned weight: %.4f (expected ~%.1f)\n", lr.Weights[0], slope)
	fmt.Printf("  Learned bias:   %.4f (expected ~0.0)\n", lr.Bias)
	fmt.Printf("  R²:  %.4f\n", r2)
	fmt.Printf("  MSE: %.4f\n", mse)
	fmt.Printf("  RMSE: %.4f\n", rmse)
	fmt.Printf("  MAE: %.4f\n", mae)

	// Show a few predictions
	fmt.Println("\n  Sample predictions (first 5 test points):")
	limit := 5
	if len(xTest) < limit {
		limit = len(xTest)
	}
	for i := 0; i < limit; i++ {
		fmt.Printf("    x=%.2f  actual=%.2f  predicted=%.2f\n",
			xTest[i][0], yTest[i], preds[i])
	}
}

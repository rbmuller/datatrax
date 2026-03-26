package ml

import (
	"math/rand"
	"testing"
)

// generateFeatures creates a deterministic dataset of n samples with nFeatures dimensions.
func generateFeatures(rng *rand.Rand, n, nFeatures int) [][]float64 {
	x := make([][]float64, n)
	for i := range x {
		row := make([]float64, nFeatures)
		for j := range row {
			row[j] = rng.Float64()*10 - 5
		}
		x[i] = row
	}
	return x
}

// generateRegressionLabels creates continuous labels based on a linear combination of features plus noise.
func generateRegressionLabels(rng *rand.Rand, x [][]float64) []float64 {
	y := make([]float64, len(x))
	for i, row := range x {
		sum := 0.0
		for j, v := range row {
			sum += float64(j+1) * v
		}
		y[i] = sum + rng.NormFloat64()*0.1
	}
	return y
}

// generateBinaryLabels creates binary labels (0 or 1) based on the sign of the first feature.
func generateBinaryLabels(x [][]float64) []float64 {
	y := make([]float64, len(x))
	for i, row := range x {
		if row[0]+row[1] > 0 {
			y[i] = 1
		}
	}
	return y
}

// generateMultiClassLabels creates labels with nClasses categories.
func generateMultiClassLabels(rng *rand.Rand, n, nClasses int) []float64 {
	y := make([]float64, n)
	for i := range y {
		y[i] = float64(rng.Intn(nClasses))
	}
	return y
}

const (
	benchSamples    = 1000
	benchFeatures   = 10
	benchPredictN   = 100
	benchSeed       = 42
	benchClasses    = 3
	benchKMeansK    = 5
)

// --- Linear Regression ---

func BenchmarkLinearRegressionFit(b *testing.B) {
	rng := rand.New(rand.NewSource(benchSeed))
	x := generateFeatures(rng, benchSamples, benchFeatures)
	y := generateRegressionLabels(rng, x)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		lr := NewLinearRegressionConfig(0.001, 100)
		lr.Fit(x, y)
	}
}

func BenchmarkLinearRegressionPredict(b *testing.B) {
	rng := rand.New(rand.NewSource(benchSeed))
	x := generateFeatures(rng, benchSamples, benchFeatures)
	y := generateRegressionLabels(rng, x)
	xPred := generateFeatures(rng, benchPredictN, benchFeatures)

	lr := NewLinearRegressionConfig(0.001, 100)
	lr.Fit(x, y)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		lr.Predict(xPred)
	}
}

// --- Logistic Regression ---

func BenchmarkLogisticRegressionFit(b *testing.B) {
	rng := rand.New(rand.NewSource(benchSeed))
	x := generateFeatures(rng, benchSamples, benchFeatures)
	y := generateBinaryLabels(x)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		lr := &LogisticRegression{
			LearningRate: 0.001,
			Epochs:       100,
			Threshold:    0.5,
		}
		lr.Fit(x, y)
	}
}

func BenchmarkLogisticRegressionPredict(b *testing.B) {
	rng := rand.New(rand.NewSource(benchSeed))
	x := generateFeatures(rng, benchSamples, benchFeatures)
	y := generateBinaryLabels(x)
	xPred := generateFeatures(rng, benchPredictN, benchFeatures)

	lr := &LogisticRegression{
		LearningRate: 0.001,
		Epochs:       100,
		Threshold:    0.5,
	}
	lr.Fit(x, y)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		lr.Predict(xPred)
	}
}

// --- KNN ---

func BenchmarkKNNPredict(b *testing.B) {
	rng := rand.New(rand.NewSource(benchSeed))
	x := generateFeatures(rng, benchSamples, benchFeatures)
	y := generateBinaryLabels(x)
	xPred := generateFeatures(rng, benchPredictN, benchFeatures)

	knn := NewKNN(KNNConfig{K: 5})
	knn.Fit(x, y)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		knn.Predict(xPred)
	}
}

// --- KMeans ---

func BenchmarkKMeansFit(b *testing.B) {
	rng := rand.New(rand.NewSource(benchSeed))
	x := generateFeatures(rng, benchSamples, benchFeatures)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		km := NewKMeans(KMeansConfig{K: benchKMeansK, MaxIter: 50})
		km.Fit(x)
	}
}

// --- Decision Tree ---

func BenchmarkDecisionTreeFit(b *testing.B) {
	rng := rand.New(rand.NewSource(benchSeed))
	x := generateFeatures(rng, benchSamples, benchFeatures)
	y := generateMultiClassLabels(rng, benchSamples, benchClasses)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dt := NewDecisionTree(DecisionTreeConfig{MaxDepth: 8, MinSamples: 5})
		dt.Fit(x, y)
	}
}

func BenchmarkDecisionTreePredict(b *testing.B) {
	rng := rand.New(rand.NewSource(benchSeed))
	x := generateFeatures(rng, benchSamples, benchFeatures)
	y := generateMultiClassLabels(rng, benchSamples, benchClasses)
	xPred := generateFeatures(rng, benchPredictN, benchFeatures)

	dt := NewDecisionTree(DecisionTreeConfig{MaxDepth: 8, MinSamples: 5})
	dt.Fit(x, y)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dt.Predict(xPred)
	}
}

// --- Gaussian Naive Bayes ---

func BenchmarkGaussianNBFit(b *testing.B) {
	rng := rand.New(rand.NewSource(benchSeed))
	x := generateFeatures(rng, benchSamples, benchFeatures)
	y := generateMultiClassLabels(rng, benchSamples, benchClasses)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		nb := NewGaussianNB()
		nb.Fit(x, y)
	}
}

func BenchmarkGaussianNBPredict(b *testing.B) {
	rng := rand.New(rand.NewSource(benchSeed))
	x := generateFeatures(rng, benchSamples, benchFeatures)
	y := generateMultiClassLabels(rng, benchSamples, benchClasses)
	xPred := generateFeatures(rng, benchPredictN, benchFeatures)

	nb := NewGaussianNB()
	nb.Fit(x, y)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		nb.Predict(xPred)
	}
}

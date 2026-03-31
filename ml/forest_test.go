package ml

import (
	"math"
	"math/rand"
	"testing"
)

func TestRandomForestBasicClassification(t *testing.T) {
	rng := rand.New(rand.NewSource(42))

	// 3 clearly separable classes based on feature values
	n := 150
	x := make([][]float64, n)
	y := make([]float64, n)
	for i := 0; i < 50; i++ {
		x[i] = []float64{rng.NormFloat64()*0.5 + 0, rng.NormFloat64()*0.5 + 0}
		y[i] = 0
	}
	for i := 50; i < 100; i++ {
		x[i] = []float64{rng.NormFloat64()*0.5 + 5, rng.NormFloat64()*0.5 + 5}
		y[i] = 1
	}
	for i := 100; i < 150; i++ {
		x[i] = []float64{rng.NormFloat64()*0.5 + 10, rng.NormFloat64()*0.5 + 0}
		y[i] = 2
	}

	rf := NewRandomForest(RandomForestConfig{
		NTrees:   50,
		MaxDepth: 8,
		Seed:     42,
	})
	rf.Fit(x, y)
	preds := rf.Predict(x)
	acc := Accuracy(y, preds)
	if acc < 0.85 {
		t.Errorf("RandomForest accuracy = %f, want > 0.85", acc)
	}
}

func TestRandomForestFeatureImportance(t *testing.T) {
	rng := rand.New(rand.NewSource(42))

	// Classification depends mainly on feature 0
	n := 200
	x := make([][]float64, n)
	y := make([]float64, n)
	for i := 0; i < n; i++ {
		f0 := rng.Float64() * 10
		f1 := rng.Float64() * 10 // noise
		f2 := rng.Float64() * 10 // noise
		x[i] = []float64{f0, f1, f2}
		if f0 < 5 {
			y[i] = 0
		} else {
			y[i] = 1
		}
	}

	rf := NewRandomForest(RandomForestConfig{
		NTrees:   30,
		MaxDepth: 5,
		Seed:     42,
	})
	rf.Fit(x, y)

	imp := rf.FeatureImportance()
	if len(imp) != 3 {
		t.Fatalf("FeatureImportance length = %d, want 3", len(imp))
	}

	// Importance should sum to approximately 1.0
	sum := 0.0
	for _, v := range imp {
		sum += v
	}
	if math.Abs(sum-1.0) > 0.01 {
		t.Errorf("FeatureImportance sum = %f, want ~1.0", sum)
	}
}

func TestRandomForestPredictProbability(t *testing.T) {
	rng := rand.New(rand.NewSource(42))

	n := 100
	x := make([][]float64, n)
	y := make([]float64, n)
	for i := 0; i < n/2; i++ {
		x[i] = []float64{rng.NormFloat64()*0.5 - 3, rng.NormFloat64() * 0.5}
		y[i] = 0
	}
	for i := n / 2; i < n; i++ {
		x[i] = []float64{rng.NormFloat64()*0.5 + 3, rng.NormFloat64() * 0.5}
		y[i] = 1
	}

	rf := NewRandomForest(RandomForestConfig{
		NTrees:   20,
		MaxDepth: 5,
		Seed:     42,
	})
	rf.Fit(x, y)
	probs := rf.PredictProbability(x)

	if len(probs) != n {
		t.Fatalf("PredictProbability rows = %d, want %d", len(probs), n)
	}

	for i, prob := range probs {
		sum := 0.0
		for _, p := range prob {
			sum += p
		}
		if math.Abs(sum-1.0) > 1e-9 {
			t.Errorf("PredictProbability sample %d sum = %f, want 1.0", i, sum)
			break
		}
	}
}

func TestRandomForestOOBScore(t *testing.T) {
	rng := rand.New(rand.NewSource(42))

	// Well-separated classes for a good OOB score
	n := 200
	x := make([][]float64, n)
	y := make([]float64, n)
	for i := 0; i < n/2; i++ {
		x[i] = []float64{rng.NormFloat64()*0.5 - 5, rng.NormFloat64() * 0.5}
		y[i] = 0
	}
	for i := n / 2; i < n; i++ {
		x[i] = []float64{rng.NormFloat64()*0.5 + 5, rng.NormFloat64() * 0.5}
		y[i] = 1
	}

	rf := NewRandomForest(RandomForestConfig{
		NTrees:   50,
		MaxDepth: 8,
		Seed:     42,
	})
	rf.Fit(x, y)
	oob := rf.OOBScore(x, y)

	if oob < 0.5 {
		t.Errorf("OOBScore = %f, want > 0.5", oob)
	}
	if oob > 1.0 {
		t.Errorf("OOBScore = %f, want <= 1.0", oob)
	}
}

func TestRandomForestDefaultConfig(t *testing.T) {
	rf := NewRandomForest(RandomForestConfig{})

	if rf.NTrees != 100 {
		t.Errorf("Default NTrees = %d, want 100", rf.NTrees)
	}
	if rf.MaxDepth != 10 {
		t.Errorf("Default MaxDepth = %d, want 10", rf.MaxDepth)
	}
	if rf.MinSamples != 2 {
		t.Errorf("Default MinSamples = %d, want 2", rf.MinSamples)
	}
	if rf.MaxFeatures != 0 {
		t.Errorf("Default MaxFeatures = %d, want 0 (auto)", rf.MaxFeatures)
	}
	if rf.Criterion != "gini" {
		t.Errorf("Default Criterion = %s, want gini", rf.Criterion)
	}
}

package ml

import "math"

// LogisticRegression implements binary logistic regression using
// gradient descent with a sigmoid activation function.
type LogisticRegression struct {
	Weights      []float64
	Bias         float64
	LearningRate float64
	Epochs       int
	Threshold    float64
}

// NewLogisticRegression creates a LogisticRegression with default
// hyperparameters (lr=0.01, epochs=1000, threshold=0.5).
func NewLogisticRegression() *LogisticRegression {
	return &LogisticRegression{
		LearningRate: 0.01,
		Epochs:       1000,
		Threshold:    0.5,
	}
}

// Fit trains the logistic regression model on features x and binary labels y
// using gradient descent with the sigmoid function.
func (lr *LogisticRegression) Fit(x [][]float64, y []float64) {
	if len(x) == 0 {
		return
	}
	n := len(x)
	nFeatures := len(x[0])

	lr.Weights = make([]float64, nFeatures)
	lr.Bias = 0

	for epoch := 0; epoch < lr.Epochs; epoch++ {
		gradW := make([]float64, nFeatures)
		gradB := 0.0

		for i := 0; i < n; i++ {
			z := lr.Bias
			for j := 0; j < nFeatures; j++ {
				z += lr.Weights[j] * x[i][j]
			}
			pred := sigmoid(z)
			err := pred - y[i]

			for j := 0; j < nFeatures; j++ {
				gradW[j] += err * x[i][j]
			}
			gradB += err
		}

		for j := 0; j < nFeatures; j++ {
			lr.Weights[j] -= lr.LearningRate * gradW[j] / float64(n)
		}
		lr.Bias -= lr.LearningRate * gradB / float64(n)
	}
}

// Predict returns binary class predictions (0 or 1) for all samples in x.
func (lr *LogisticRegression) Predict(x [][]float64) []float64 {
	probs := lr.PredictProbability(x)
	preds := make([]float64, len(probs))
	for i, p := range probs {
		if p >= lr.Threshold {
			preds[i] = 1
		}
	}
	return preds
}

// PredictProbability returns the predicted probability of the positive class
// for all samples in x.
func (lr *LogisticRegression) PredictProbability(x [][]float64) []float64 {
	probs := make([]float64, len(x))
	for i, row := range x {
		z := lr.Bias
		for j, w := range lr.Weights {
			z += w * row[j]
		}
		probs[i] = sigmoid(z)
	}
	return probs
}

// sigmoid computes the sigmoid function: 1 / (1 + exp(-x)).
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

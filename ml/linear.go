package ml

// LinearRegression implements linear regression using gradient descent.
type LinearRegression struct {
	Weights      []float64
	Bias         float64
	LearningRate float64
	Epochs       int
}

// NewLinearRegression creates a LinearRegression with default hyperparameters
// (learning rate = 0.01, epochs = 1000).
func NewLinearRegression() *LinearRegression {
	return &LinearRegression{
		LearningRate: 0.01,
		Epochs:       1000,
	}
}

// NewLinearRegressionConfig creates a LinearRegression with custom learning
// rate and number of epochs.
func NewLinearRegressionConfig(lr float64, epochs int) *LinearRegression {
	return &LinearRegression{
		LearningRate: lr,
		Epochs:       epochs,
	}
}

// Fit trains the linear regression model on features x and labels y
// using gradient descent.
func (lr *LinearRegression) Fit(x [][]float64, y []float64) {
	if len(x) == 0 {
		return
	}
	n := len(x)
	nFeatures := len(x[0])

	lr.Weights = make([]float64, nFeatures)
	lr.Bias = 0

	for epoch := 0; epoch < lr.Epochs; epoch++ {
		// Compute gradients
		gradW := make([]float64, nFeatures)
		gradB := 0.0

		for i := 0; i < n; i++ {
			pred := lr.predictSingle(x[i])
			err := pred - y[i]
			for j := 0; j < nFeatures; j++ {
				gradW[j] += err * x[i][j]
			}
			gradB += err
		}

		// Update weights
		for j := 0; j < nFeatures; j++ {
			lr.Weights[j] -= lr.LearningRate * gradW[j] / float64(n)
		}
		lr.Bias -= lr.LearningRate * gradB / float64(n)
	}
}

// FitNormalEquation trains the linear regression model using the normal
// equation: w = (X^T X)^{-1} X^T y. Falls back to gradient descent if
// the matrix is singular.
func (lr *LinearRegression) FitNormalEquation(x [][]float64, y []float64) {
	if len(x) == 0 {
		return
	}
	n := len(x)
	p := len(x[0]) + 1 // +1 for bias term

	// Build augmented matrix X with bias column
	xAug := make([][]float64, n)
	for i := range x {
		row := make([]float64, p)
		row[0] = 1.0 // bias
		copy(row[1:], x[i])
		xAug[i] = row
	}

	// Compute X^T X (p x p)
	xtx := make([][]float64, p)
	for i := range xtx {
		xtx[i] = make([]float64, p)
		for j := range xtx[i] {
			sum := 0.0
			for k := 0; k < n; k++ {
				sum += xAug[k][i] * xAug[k][j]
			}
			xtx[i][j] = sum
		}
	}

	// Compute X^T y (p x 1)
	xty := make([]float64, p)
	for i := 0; i < p; i++ {
		sum := 0.0
		for k := 0; k < n; k++ {
			sum += xAug[k][i] * y[k]
		}
		xty[i] = sum
	}

	// Solve via Gauss-Jordan elimination
	// Build augmented matrix [XtX | Xty]
	aug := make([][]float64, p)
	for i := range aug {
		aug[i] = make([]float64, p+1)
		copy(aug[i], xtx[i])
		aug[i][p] = xty[i]
	}

	for i := 0; i < p; i++ {
		// Partial pivoting
		maxRow := i
		for k := i + 1; k < p; k++ {
			if abs(aug[k][i]) > abs(aug[maxRow][i]) {
				maxRow = k
			}
		}
		aug[i], aug[maxRow] = aug[maxRow], aug[i]

		pivot := aug[i][i]
		if abs(pivot) < 1e-12 {
			// Singular matrix, fall back to gradient descent
			lr.Fit(x, y)
			return
		}

		for j := i; j <= p; j++ {
			aug[i][j] /= pivot
		}

		for k := 0; k < p; k++ {
			if k == i {
				continue
			}
			factor := aug[k][i]
			for j := i; j <= p; j++ {
				aug[k][j] -= factor * aug[i][j]
			}
		}
	}

	// Extract solution
	w := make([]float64, p)
	for i := 0; i < p; i++ {
		w[i] = aug[i][p]
	}

	lr.Bias = w[0]
	lr.Weights = w[1:]
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

// Predict returns predictions for all samples in x.
func (lr *LinearRegression) Predict(x [][]float64) []float64 {
	preds := make([]float64, len(x))
	for i, row := range x {
		preds[i] = lr.predictSingle(row)
	}
	return preds
}

func (lr *LinearRegression) predictSingle(x []float64) float64 {
	sum := lr.Bias
	for j, w := range lr.Weights {
		sum += w * x[j]
	}
	return sum
}

package ml

// Fold represents one fold of a cross-validation split.
type Fold struct {
	XTrain [][]float64
	XTest  [][]float64
	YTrain []float64
	YTest  []float64
}

// KFoldSplit divides the data into k folds for cross-validation.
// Each fold uses 1/k of the data as test and the remaining as training.
func KFoldSplit(x [][]float64, y []float64, k int) []Fold {
	n := len(x)
	if k <= 0 || k > n {
		k = n
	}

	foldSize := n / k
	remainder := n % k

	folds := make([]Fold, k)
	start := 0

	for i := 0; i < k; i++ {
		size := foldSize
		if i < remainder {
			size++
		}
		end := start + size

		var xTrain, xTest [][]float64
		var yTrain, yTest []float64

		for j := 0; j < n; j++ {
			if j >= start && j < end {
				xTest = append(xTest, x[j])
				yTest = append(yTest, y[j])
			} else {
				xTrain = append(xTrain, x[j])
				yTrain = append(yTrain, y[j])
			}
		}

		folds[i] = Fold{
			XTrain: xTrain,
			XTest:  xTest,
			YTrain: yTrain,
			YTest:  yTest,
		}

		start = end
	}

	return folds
}

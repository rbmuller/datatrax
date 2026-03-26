package ml

import "math"

// Accuracy computes the fraction of correct predictions.
func Accuracy(yTrue, yPred []float64) float64 {
	if len(yTrue) == 0 {
		return 0
	}
	correct := 0
	for i := range yTrue {
		if yTrue[i] == yPred[i] {
			correct++
		}
	}
	return float64(correct) / float64(len(yTrue))
}

// Precision computes precision for the given positive class.
// Precision = TP / (TP + FP).
func Precision(yTrue, yPred []float64, positiveClass float64) float64 {
	cm := confusionMatrixValues(yTrue, yPred, positiveClass)
	denom := cm.tp + cm.fp
	if denom == 0 {
		return 0
	}
	return float64(cm.tp) / float64(denom)
}

// Recall computes recall for the given positive class.
// Recall = TP / (TP + FN).
func Recall(yTrue, yPred []float64, positiveClass float64) float64 {
	cm := confusionMatrixValues(yTrue, yPred, positiveClass)
	denom := cm.tp + cm.fn
	if denom == 0 {
		return 0
	}
	return float64(cm.tp) / float64(denom)
}

// F1Score computes the F1 score (harmonic mean of precision and recall)
// for the given positive class.
func F1Score(yTrue, yPred []float64, positiveClass float64) float64 {
	p := Precision(yTrue, yPred, positiveClass)
	r := Recall(yTrue, yPred, positiveClass)
	if p+r == 0 {
		return 0
	}
	return 2 * p * r / (p + r)
}

// ConfusionMatrix returns a map with keys "tp", "fp", "tn", "fn" for
// binary classification where positive class is 1.0.
func ConfusionMatrix(yTrue, yPred []float64) map[string]int {
	cm := confusionMatrixValues(yTrue, yPred, 1.0)
	return map[string]int{
		"tp": cm.tp,
		"fp": cm.fp,
		"tn": cm.tn,
		"fn": cm.fn,
	}
}

type cmValues struct {
	tp, fp, tn, fn int
}

func confusionMatrixValues(yTrue, yPred []float64, positiveClass float64) cmValues {
	var cm cmValues
	for i := range yTrue {
		actual := yTrue[i] == positiveClass
		predicted := yPred[i] == positiveClass
		switch {
		case actual && predicted:
			cm.tp++
		case !actual && predicted:
			cm.fp++
		case actual && !predicted:
			cm.fn++
		default:
			cm.tn++
		}
	}
	return cm
}

// MSE computes the Mean Squared Error between true and predicted values.
func MSE(yTrue, yPred []float64) float64 {
	if len(yTrue) == 0 {
		return 0
	}
	sum := 0.0
	for i := range yTrue {
		diff := yTrue[i] - yPred[i]
		sum += diff * diff
	}
	return sum / float64(len(yTrue))
}

// RMSE computes the Root Mean Squared Error between true and predicted values.
func RMSE(yTrue, yPred []float64) float64 {
	return math.Sqrt(MSE(yTrue, yPred))
}

// MAE computes the Mean Absolute Error between true and predicted values.
func MAE(yTrue, yPred []float64) float64 {
	if len(yTrue) == 0 {
		return 0
	}
	sum := 0.0
	for i := range yTrue {
		sum += math.Abs(yTrue[i] - yPred[i])
	}
	return sum / float64(len(yTrue))
}

// R2Score computes the R-squared (coefficient of determination) metric.
// R2 = 1 - SS_res / SS_tot.
func R2Score(yTrue, yPred []float64) float64 {
	if len(yTrue) == 0 {
		return 0
	}
	mean := 0.0
	for _, v := range yTrue {
		mean += v
	}
	mean /= float64(len(yTrue))

	ssRes := 0.0
	ssTot := 0.0
	for i := range yTrue {
		ssRes += (yTrue[i] - yPred[i]) * (yTrue[i] - yPred[i])
		ssTot += (yTrue[i] - mean) * (yTrue[i] - mean)
	}
	if ssTot == 0 {
		return 0
	}
	return 1 - ssRes/ssTot
}

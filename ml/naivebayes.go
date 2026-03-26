package ml

import "math"

// GaussianNB implements the Gaussian Naive Bayes classifier.
// It assumes features follow a Gaussian distribution within each class.
type GaussianNB struct {
	classes   []float64
	priors    map[float64]float64
	means     map[float64][]float64
	variances map[float64][]float64
}

// NewGaussianNB creates a new Gaussian Naive Bayes classifier.
func NewGaussianNB() *GaussianNB {
	return &GaussianNB{}
}

// Fit computes the class priors, means, and variances from the training data.
func (nb *GaussianNB) Fit(x [][]float64, y []float64) {
	if len(x) == 0 {
		return
	}
	n := len(x)
	nFeatures := len(x[0])

	// Group data by class
	classData := make(map[float64][][]float64)
	for i := range x {
		classData[y[i]] = append(classData[y[i]], x[i])
	}

	nb.classes = make([]float64, 0, len(classData))
	nb.priors = make(map[float64]float64)
	nb.means = make(map[float64][]float64)
	nb.variances = make(map[float64][]float64)

	for class, data := range classData {
		nb.classes = append(nb.classes, class)
		nb.priors[class] = float64(len(data)) / float64(n)

		means := make([]float64, nFeatures)
		variances := make([]float64, nFeatures)

		for j := 0; j < nFeatures; j++ {
			sum := 0.0
			for _, row := range data {
				sum += row[j]
			}
			means[j] = sum / float64(len(data))

			sumSq := 0.0
			for _, row := range data {
				diff := row[j] - means[j]
				sumSq += diff * diff
			}
			variances[j] = sumSq / float64(len(data))
			// Add small epsilon to avoid division by zero
			if variances[j] < 1e-12 {
				variances[j] = 1e-12
			}
		}

		nb.means[class] = means
		nb.variances[class] = variances
	}
}

// Predict returns predicted class labels for all samples in x.
func (nb *GaussianNB) Predict(x [][]float64) []float64 {
	probs := nb.PredictProbability(x)
	preds := make([]float64, len(x))
	for i, classProbs := range probs {
		bestClass := nb.classes[0]
		bestProb := -math.MaxFloat64
		for j, class := range nb.classes {
			if classProbs[j] > bestProb {
				bestProb = classProbs[j]
				bestClass = class
			}
		}
		preds[i] = bestClass
	}
	return preds
}

// PredictProbability returns the log-probability of each class for all
// samples in x. Each inner slice corresponds to the classes in the
// same order as they were encountered during Fit.
func (nb *GaussianNB) PredictProbability(x [][]float64) [][]float64 {
	result := make([][]float64, len(x))
	for i, sample := range x {
		logProbs := make([]float64, len(nb.classes))
		for j, class := range nb.classes {
			logProb := math.Log(nb.priors[class])
			means := nb.means[class]
			variances := nb.variances[class]
			for f := range sample {
				logProb += gaussianLogPDF(sample[f], means[f], variances[f])
			}
			logProbs[j] = logProb
		}
		result[i] = logProbs
	}
	return result
}

// gaussianLogPDF computes the log of the Gaussian probability density function.
func gaussianLogPDF(x, mean, variance float64) float64 {
	return -0.5*math.Log(2*math.Pi*variance) - 0.5*(x-mean)*(x-mean)/variance
}

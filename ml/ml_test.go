package ml

import (
	"math"
	"math/rand"
	"testing"
)

// --- Linear Regression ---

func TestLinearRegression(t *testing.T) {
	// y = 2x + 1
	rand.Seed(42)
	x := make([][]float64, 100)
	y := make([]float64, 100)
	for i := 0; i < 100; i++ {
		xi := float64(i) / 10.0
		x[i] = []float64{xi}
		y[i] = 2*xi + 1
	}

	// Scale features for stable gradient descent
	xScaled := MinMaxScale(x)

	lr := NewLinearRegressionConfig(0.1, 2000)
	lr.Fit(xScaled, y)

	preds := lr.Predict(xScaled)
	r2 := R2Score(y, preds)
	if r2 < 0.95 {
		t.Errorf("LinearRegression R2 = %f, want > 0.95", r2)
	}
}

func TestLinearRegressionNormalEquation(t *testing.T) {
	// y = 2x + 1
	x := make([][]float64, 100)
	y := make([]float64, 100)
	for i := 0; i < 100; i++ {
		xi := float64(i) / 10.0
		x[i] = []float64{xi}
		y[i] = 2*xi + 1
	}

	lr := NewLinearRegression()
	lr.FitNormalEquation(x, y)

	preds := lr.Predict(x)
	r2 := R2Score(y, preds)
	if r2 < 0.99 {
		t.Errorf("LinearRegression (normal eq) R2 = %f, want > 0.99", r2)
	}

	// Check weights are close to expected
	if math.Abs(lr.Weights[0]-2.0) > 0.01 {
		t.Errorf("Weight = %f, want ~2.0", lr.Weights[0])
	}
	if math.Abs(lr.Bias-1.0) > 0.01 {
		t.Errorf("Bias = %f, want ~1.0", lr.Bias)
	}
}

// --- Logistic Regression ---

func TestLogisticRegression(t *testing.T) {
	rand.Seed(42)

	// Linearly separable data: class 0 centered at -2, class 1 at +2
	n := 200
	x := make([][]float64, n)
	y := make([]float64, n)
	for i := 0; i < n/2; i++ {
		x[i] = []float64{-2 + rand.NormFloat64()*0.5}
		y[i] = 0
	}
	for i := n / 2; i < n; i++ {
		x[i] = []float64{2 + rand.NormFloat64()*0.5}
		y[i] = 1
	}

	lr := NewLogisticRegression()
	lr.LearningRate = 0.1
	lr.Epochs = 2000
	lr.Fit(x, y)

	preds := lr.Predict(x)
	acc := Accuracy(y, preds)
	if acc < 0.9 {
		t.Errorf("LogisticRegression accuracy = %f, want > 0.9", acc)
	}

	// Test probabilities
	probs := lr.PredictProbability(x)
	if len(probs) != n {
		t.Errorf("PredictProbability returned %d, want %d", len(probs), n)
	}
	for _, p := range probs {
		if p < 0 || p > 1 {
			t.Errorf("Probability %f out of range [0, 1]", p)
			break
		}
	}
}

// --- KNN ---

func TestKNN(t *testing.T) {
	rand.Seed(42)

	// Two clusters
	n := 100
	x := make([][]float64, n)
	y := make([]float64, n)
	for i := 0; i < n/2; i++ {
		x[i] = []float64{rand.NormFloat64()*0.5 + 0, rand.NormFloat64()*0.5 + 0}
		y[i] = 0
	}
	for i := n / 2; i < n; i++ {
		x[i] = []float64{rand.NormFloat64()*0.5 + 5, rand.NormFloat64()*0.5 + 5}
		y[i] = 1
	}

	knn := NewKNN(KNNConfig{K: 5, Distance: "euclidean"})
	knn.Fit(x, y)
	preds := knn.Predict(x)
	acc := Accuracy(y, preds)
	if acc < 0.9 {
		t.Errorf("KNN accuracy = %f, want > 0.9", acc)
	}
}

func TestKNNManhattan(t *testing.T) {
	knn := NewKNN(KNNConfig{K: 3, Distance: "manhattan"})
	x := [][]float64{{0, 0}, {1, 1}, {10, 10}, {11, 11}}
	y := []float64{0, 0, 1, 1}
	knn.Fit(x, y)
	preds := knn.Predict([][]float64{{0.5, 0.5}, {10.5, 10.5}})
	if preds[0] != 0 {
		t.Errorf("KNN Manhattan pred[0] = %f, want 0", preds[0])
	}
	if preds[1] != 1 {
		t.Errorf("KNN Manhattan pred[1] = %f, want 1", preds[1])
	}
}

func TestKNNDefaults(t *testing.T) {
	knn := NewKNN(KNNConfig{})
	if knn.K != 5 {
		t.Errorf("Default K = %d, want 5", knn.K)
	}
	if knn.Distance != "euclidean" {
		t.Errorf("Default Distance = %s, want euclidean", knn.Distance)
	}
}

// --- K-Means ---

func TestKMeans(t *testing.T) {
	rand.Seed(42)

	// 3 well-separated clusters
	x := make([][]float64, 0, 150)
	for i := 0; i < 50; i++ {
		x = append(x, []float64{rand.NormFloat64()*0.3 + 0, rand.NormFloat64()*0.3 + 0})
	}
	for i := 0; i < 50; i++ {
		x = append(x, []float64{rand.NormFloat64()*0.3 + 10, rand.NormFloat64()*0.3 + 10})
	}
	for i := 0; i < 50; i++ {
		x = append(x, []float64{rand.NormFloat64()*0.3 + 20, rand.NormFloat64()*0.3 + 0})
	}

	km := NewKMeans(KMeansConfig{K: 3, MaxIter: 100})
	km.Fit(x)
	labels := km.Predict(x)

	// Check that points in the same original group get the same label
	label0 := labels[0]
	label1 := labels[50]
	label2 := labels[100]

	// All three clusters should have different labels
	if label0 == label1 || label0 == label2 || label1 == label2 {
		t.Errorf("KMeans did not separate clusters: labels = %d, %d, %d", label0, label1, label2)
	}

	// Check that all points in cluster 0 have the same label
	for i := 0; i < 50; i++ {
		if labels[i] != label0 {
			t.Errorf("KMeans cluster 0 point %d has label %d, want %d", i, labels[i], label0)
			break
		}
	}
	for i := 50; i < 100; i++ {
		if labels[i] != label1 {
			t.Errorf("KMeans cluster 1 point %d has label %d, want %d", i, labels[i], label1)
			break
		}
	}
	for i := 100; i < 150; i++ {
		if labels[i] != label2 {
			t.Errorf("KMeans cluster 2 point %d has label %d, want %d", i, labels[i], label2)
			break
		}
	}

	// Check inertia is computed and positive
	if km.Inertia() <= 0 {
		t.Errorf("KMeans inertia = %f, want > 0", km.Inertia())
	}
}

// --- Decision Tree ---

func TestDecisionTree(t *testing.T) {
	// Simple classification: class 0 for x1 < 5, class 1 for x1 >= 5
	x := [][]float64{
		{0, 1}, {1, 2}, {2, 1}, {3, 3}, {4, 2},
		{5, 1}, {6, 2}, {7, 3}, {8, 1}, {9, 2},
	}
	y := []float64{0, 0, 0, 0, 0, 1, 1, 1, 1, 1}

	dt := NewDecisionTree(DecisionTreeConfig{MaxDepth: 5, MinSamples: 1, Criterion: "gini"})
	dt.Fit(x, y)
	preds := dt.Predict(x)
	acc := Accuracy(y, preds)
	if acc < 0.9 {
		t.Errorf("DecisionTree accuracy = %f, want > 0.9", acc)
	}

	// Feature importance should be defined
	imp := dt.FeatureImportance()
	if len(imp) != 2 {
		t.Errorf("FeatureImportance length = %d, want 2", len(imp))
	}
	// Feature 0 should be more important than feature 1
	if imp[0] < imp[1] {
		t.Errorf("Feature 0 importance (%f) should be > feature 1 (%f)", imp[0], imp[1])
	}
}

func TestDecisionTreeEntropy(t *testing.T) {
	x := [][]float64{{0}, {1}, {2}, {3}, {4}, {5}}
	y := []float64{0, 0, 0, 1, 1, 1}

	dt := NewDecisionTree(DecisionTreeConfig{Criterion: "entropy"})
	dt.Fit(x, y)
	preds := dt.Predict(x)
	acc := Accuracy(y, preds)
	if acc < 1.0 {
		t.Errorf("DecisionTree (entropy) accuracy = %f, want 1.0", acc)
	}
}

// --- Naive Bayes ---

func TestGaussianNB(t *testing.T) {
	rand.Seed(42)

	// Two Gaussian distributed classes
	n := 200
	x := make([][]float64, n)
	y := make([]float64, n)
	for i := 0; i < n/2; i++ {
		x[i] = []float64{rand.NormFloat64()*1 + 0, rand.NormFloat64()*1 + 0}
		y[i] = 0
	}
	for i := n / 2; i < n; i++ {
		x[i] = []float64{rand.NormFloat64()*1 + 5, rand.NormFloat64()*1 + 5}
		y[i] = 1
	}

	nb := NewGaussianNB()
	nb.Fit(x, y)
	preds := nb.Predict(x)
	acc := Accuracy(y, preds)
	if acc < 0.9 {
		t.Errorf("GaussianNB accuracy = %f, want > 0.9", acc)
	}

	// Test PredictProbability returns correct dimensions
	probs := nb.PredictProbability(x)
	if len(probs) != n {
		t.Errorf("PredictProbability rows = %d, want %d", len(probs), n)
	}
	if len(probs[0]) != 2 {
		t.Errorf("PredictProbability cols = %d, want 2", len(probs[0]))
	}
}

// --- Metrics ---

func TestAccuracy(t *testing.T) {
	yTrue := []float64{1, 1, 0, 0, 1}
	yPred := []float64{1, 0, 0, 0, 1}
	acc := Accuracy(yTrue, yPred)
	expected := 4.0 / 5.0
	if math.Abs(acc-expected) > 1e-9 {
		t.Errorf("Accuracy = %f, want %f", acc, expected)
	}
}

func TestPrecisionRecallF1(t *testing.T) {
	yTrue := []float64{1, 1, 1, 0, 0}
	yPred := []float64{1, 1, 0, 0, 1}

	p := Precision(yTrue, yPred, 1.0)
	// TP=2, FP=1 -> precision = 2/3
	if math.Abs(p-2.0/3.0) > 1e-9 {
		t.Errorf("Precision = %f, want %f", p, 2.0/3.0)
	}

	r := Recall(yTrue, yPred, 1.0)
	// TP=2, FN=1 -> recall = 2/3
	if math.Abs(r-2.0/3.0) > 1e-9 {
		t.Errorf("Recall = %f, want %f", r, 2.0/3.0)
	}

	f1 := F1Score(yTrue, yPred, 1.0)
	expectedF1 := 2 * p * r / (p + r)
	if math.Abs(f1-expectedF1) > 1e-9 {
		t.Errorf("F1Score = %f, want %f", f1, expectedF1)
	}
}

func TestConfusionMatrix(t *testing.T) {
	yTrue := []float64{1, 1, 0, 0}
	yPred := []float64{1, 0, 0, 1}
	cm := ConfusionMatrix(yTrue, yPred)
	if cm["tp"] != 1 || cm["fn"] != 1 || cm["tn"] != 1 || cm["fp"] != 1 {
		t.Errorf("ConfusionMatrix = %v, want tp=1 fn=1 tn=1 fp=1", cm)
	}
}

func TestMSE(t *testing.T) {
	yTrue := []float64{1, 2, 3}
	yPred := []float64{1.1, 2.1, 3.1}
	mse := MSE(yTrue, yPred)
	expected := (0.01 + 0.01 + 0.01) / 3
	if math.Abs(mse-expected) > 1e-9 {
		t.Errorf("MSE = %f, want %f", mse, expected)
	}
}

func TestRMSE(t *testing.T) {
	yTrue := []float64{1, 2, 3}
	yPred := []float64{1.1, 2.1, 3.1}
	rmse := RMSE(yTrue, yPred)
	expected := math.Sqrt((0.01 + 0.01 + 0.01) / 3)
	if math.Abs(rmse-expected) > 1e-9 {
		t.Errorf("RMSE = %f, want %f", rmse, expected)
	}
}

func TestMAE(t *testing.T) {
	yTrue := []float64{1, 2, 3}
	yPred := []float64{1.5, 2.5, 3.5}
	mae := MAE(yTrue, yPred)
	if math.Abs(mae-0.5) > 1e-9 {
		t.Errorf("MAE = %f, want 0.5", mae)
	}
}

func TestR2Score(t *testing.T) {
	yTrue := []float64{1, 2, 3, 4, 5}
	yPred := []float64{1, 2, 3, 4, 5}
	r2 := R2Score(yTrue, yPred)
	if math.Abs(r2-1.0) > 1e-9 {
		t.Errorf("R2Score (perfect) = %f, want 1.0", r2)
	}

	// Bad predictions
	yPredBad := []float64{5, 4, 3, 2, 1}
	r2Bad := R2Score(yTrue, yPredBad)
	if r2Bad > 0 {
		t.Errorf("R2Score (reversed) = %f, want <= 0", r2Bad)
	}
}

// --- Preprocessing ---

func TestMinMaxScale(t *testing.T) {
	data := [][]float64{
		{1, 10},
		{2, 20},
		{3, 30},
		{4, 40},
		{5, 50},
	}
	scaled := MinMaxScale(data)

	for _, row := range scaled {
		for _, v := range row {
			if v < 0 || v > 1 {
				t.Errorf("MinMaxScale value %f out of range [0, 1]", v)
			}
		}
	}

	// Check first and last
	if scaled[0][0] != 0 {
		t.Errorf("MinMaxScale min = %f, want 0", scaled[0][0])
	}
	if scaled[4][0] != 1 {
		t.Errorf("MinMaxScale max = %f, want 1", scaled[4][0])
	}
}

func TestStandardScale(t *testing.T) {
	data := [][]float64{
		{1, 10},
		{2, 20},
		{3, 30},
		{4, 40},
		{5, 50},
	}
	scaled := StandardScale(data)

	// Mean should be ~0
	for col := 0; col < 2; col++ {
		mean := MeanOfColumn(scaled, col)
		if math.Abs(mean) > 1e-9 {
			t.Errorf("StandardScale mean of col %d = %f, want ~0", col, mean)
		}
	}

	// Std should be ~1
	for col := 0; col < 2; col++ {
		std := StdDevOfColumn(scaled, col)
		if math.Abs(std-1.0) > 1e-9 {
			t.Errorf("StandardScale std of col %d = %f, want ~1", col, std)
		}
	}
}

func TestColumnStats(t *testing.T) {
	data := [][]float64{{1}, {2}, {3}, {4}, {5}}

	if MinOfColumn(data, 0) != 1 {
		t.Error("MinOfColumn failed")
	}
	if MaxOfColumn(data, 0) != 5 {
		t.Error("MaxOfColumn failed")
	}
	if MeanOfColumn(data, 0) != 3 {
		t.Error("MeanOfColumn failed")
	}
	expectedStd := math.Sqrt(2.0) // population std of {1,2,3,4,5}
	if math.Abs(StdDevOfColumn(data, 0)-expectedStd) > 1e-9 {
		t.Errorf("StdDevOfColumn = %f, want %f", StdDevOfColumn(data, 0), expectedStd)
	}
}

// --- Dataset ---

func TestDatasetSplit(t *testing.T) {
	rand.Seed(42)
	n := 100
	x := make([][]float64, n)
	y := make([]float64, n)
	for i := 0; i < n; i++ {
		x[i] = []float64{float64(i)}
		y[i] = float64(i)
	}

	ds := NewDataset(x, y)
	xTrain, xTest, yTrain, yTest := ds.Split(0.8)

	if len(xTrain) != 80 {
		t.Errorf("Train size = %d, want 80", len(xTrain))
	}
	if len(xTest) != 20 {
		t.Errorf("Test size = %d, want 20", len(xTest))
	}
	if len(yTrain) != 80 {
		t.Errorf("yTrain size = %d, want 80", len(yTrain))
	}
	if len(yTest) != 20 {
		t.Errorf("yTest size = %d, want 20", len(yTest))
	}
}

func TestDatasetShape(t *testing.T) {
	x := [][]float64{{1, 2}, {3, 4}, {5, 6}}
	y := []float64{0, 1, 0}
	ds := NewDataset(x, y)
	rows, cols := ds.Shape()
	if rows != 3 || cols != 2 {
		t.Errorf("Shape = (%d, %d), want (3, 2)", rows, cols)
	}
}

func TestDatasetShuffle(t *testing.T) {
	rand.Seed(42)
	x := [][]float64{{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}}
	y := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	ds := NewDataset(x, y)
	ds.Shuffle()

	// After shuffle, order should differ (with very high probability)
	inOrder := true
	for i := range y {
		if ds.Y[i] != float64(i+1) {
			inOrder = false
			break
		}
	}
	if inOrder {
		t.Error("Shuffle did not change order (statistically improbable)")
	}

	// But same values should be present
	sum := 0.0
	for _, v := range ds.Y {
		sum += v
	}
	if sum != 55 {
		t.Errorf("Shuffle changed values: sum = %f, want 55", sum)
	}
}

// --- Cross Validation ---

func TestKFoldSplit(t *testing.T) {
	x := make([][]float64, 10)
	y := make([]float64, 10)
	for i := 0; i < 10; i++ {
		x[i] = []float64{float64(i)}
		y[i] = float64(i)
	}

	folds := KFoldSplit(x, y, 5)
	if len(folds) != 5 {
		t.Errorf("KFoldSplit returned %d folds, want 5", len(folds))
	}

	for i, fold := range folds {
		if len(fold.XTest) != 2 {
			t.Errorf("Fold %d test size = %d, want 2", i, len(fold.XTest))
		}
		if len(fold.XTrain) != 8 {
			t.Errorf("Fold %d train size = %d, want 8", i, len(fold.XTrain))
		}
	}
}

func TestKFoldSplitUneven(t *testing.T) {
	x := make([][]float64, 7)
	y := make([]float64, 7)
	for i := 0; i < 7; i++ {
		x[i] = []float64{float64(i)}
		y[i] = float64(i)
	}

	folds := KFoldSplit(x, y, 3)
	if len(folds) != 3 {
		t.Errorf("KFoldSplit returned %d folds, want 3", len(folds))
	}

	// Total test samples across folds should equal n
	totalTest := 0
	for _, fold := range folds {
		totalTest += len(fold.XTest)
	}
	if totalTest != 7 {
		t.Errorf("Total test samples = %d, want 7", totalTest)
	}
}

// --- Sigmoid ---

func TestSigmoid(t *testing.T) {
	if math.Abs(sigmoid(0)-0.5) > 1e-9 {
		t.Errorf("sigmoid(0) = %f, want 0.5", sigmoid(0))
	}
	if sigmoid(100) < 0.99 {
		t.Errorf("sigmoid(100) = %f, want ~1.0", sigmoid(100))
	}
	if sigmoid(-100) > 0.01 {
		t.Errorf("sigmoid(-100) = %f, want ~0.0", sigmoid(-100))
	}
}

// --- Distance functions ---

func TestEuclideanDistance(t *testing.T) {
	a := []float64{0, 0}
	b := []float64{3, 4}
	d := euclideanDistance(a, b)
	if math.Abs(d-5.0) > 1e-9 {
		t.Errorf("euclideanDistance = %f, want 5.0", d)
	}
}

func TestManhattanDistance(t *testing.T) {
	a := []float64{0, 0}
	b := []float64{3, 4}
	d := manhattanDistance(a, b)
	if math.Abs(d-7.0) > 1e-9 {
		t.Errorf("manhattanDistance = %f, want 7.0", d)
	}
}

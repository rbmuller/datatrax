package ml

import (
	"math"
	"math/rand"
)

// RandomForest implements an ensemble of decision trees trained on
// bootstrapped samples with random feature subsets (bagging).
type RandomForest struct {
	NTrees      int
	MaxDepth    int
	MinSamples  int
	MaxFeatures int
	Criterion   string
	Seed        int64

	trees          []*DecisionTree
	featureIndices [][]int
	nFeatures      int
	classes        []float64
	oobPredictions map[int]map[float64]int // sample index -> class -> vote count
}

// RandomForestConfig holds configuration for creating a RandomForest.
type RandomForestConfig struct {
	NTrees      int
	MaxDepth    int
	MinSamples  int
	MaxFeatures int
	Criterion   string
	Seed        int64
}

// NewRandomForest creates a new RandomForest with the given configuration.
// Defaults: NTrees=100, MaxDepth=10, MinSamples=2, MaxFeatures=sqrt(n_features), Criterion="gini".
func NewRandomForest(config RandomForestConfig) *RandomForest {
	nTrees := config.NTrees
	if nTrees <= 0 {
		nTrees = 100
	}
	maxDepth := config.MaxDepth
	if maxDepth <= 0 {
		maxDepth = 10
	}
	minSamples := config.MinSamples
	if minSamples <= 0 {
		minSamples = 2
	}
	criterion := config.Criterion
	if criterion == "" {
		criterion = "gini"
	}
	return &RandomForest{
		NTrees:      nTrees,
		MaxDepth:    maxDepth,
		MinSamples:  minSamples,
		MaxFeatures: config.MaxFeatures, // 0 means auto (sqrt) at Fit time
		Criterion:   criterion,
		Seed:        config.Seed,
	}
}

// Fit builds the random forest from training features x and labels y.
// Each tree is trained on a bootstrapped sample with a random feature subset.
func (rf *RandomForest) Fit(x [][]float64, y []float64) {
	if len(x) == 0 {
		return
	}

	n := len(x)
	rf.nFeatures = len(x[0])

	maxFeatures := rf.MaxFeatures
	if maxFeatures <= 0 {
		maxFeatures = int(math.Sqrt(float64(rf.nFeatures)))
		if maxFeatures < 1 {
			maxFeatures = 1
		}
	}
	if maxFeatures > rf.nFeatures {
		maxFeatures = rf.nFeatures
	}

	// Collect unique classes
	classSet := classCounts(y)
	rf.classes = make([]float64, 0, len(classSet))
	for c := range classSet {
		rf.classes = append(rf.classes, c)
	}

	rf.trees = make([]*DecisionTree, rf.NTrees)
	rf.featureIndices = make([][]int, rf.NTrees)
	rf.oobPredictions = make(map[int]map[float64]int)

	rng := rand.New(rand.NewSource(rf.Seed))

	for t := 0; t < rf.NTrees; t++ {
		// Bootstrap sampling: sample n rows with replacement
		bootIdx := make([]int, n)
		inBag := make(map[int]bool)
		for i := 0; i < n; i++ {
			idx := rng.Intn(n)
			bootIdx[i] = idx
			inBag[idx] = true
		}

		// Feature bagging: select maxFeatures random features
		perm := rng.Perm(rf.nFeatures)
		features := make([]int, maxFeatures)
		copy(features, perm[:maxFeatures])
		rf.featureIndices[t] = features

		// Build bootstrapped dataset with selected features only
		bootX := make([][]float64, n)
		bootY := make([]float64, n)
		for i, idx := range bootIdx {
			row := make([]float64, maxFeatures)
			for j, f := range features {
				row[j] = x[idx][f]
			}
			bootX[i] = row
			bootY[i] = y[idx]
		}

		// Train a decision tree on the bootstrapped data
		dt := NewDecisionTree(DecisionTreeConfig{
			MaxDepth:   rf.MaxDepth,
			MinSamples: rf.MinSamples,
			Criterion:  rf.Criterion,
		})
		dt.Fit(bootX, bootY)
		rf.trees[t] = dt

		// OOB predictions: predict samples not in the bootstrap
		for i := 0; i < n; i++ {
			if inBag[i] {
				continue
			}
			row := make([]float64, maxFeatures)
			for j, f := range features {
				row[j] = x[i][f]
			}
			pred := dt.Predict([][]float64{row})[0]
			if rf.oobPredictions[i] == nil {
				rf.oobPredictions[i] = make(map[float64]int)
			}
			rf.oobPredictions[i][pred]++
		}
	}
}

// Predict returns predicted class labels for all samples in x using majority vote.
func (rf *RandomForest) Predict(x [][]float64) []float64 {
	preds := make([]float64, len(x))
	for i, row := range x {
		votes := make(map[float64]int)
		for t, dt := range rf.trees {
			features := rf.featureIndices[t]
			subset := make([]float64, len(features))
			for j, f := range features {
				subset[j] = row[f]
			}
			pred := dt.Predict([][]float64{subset})[0]
			votes[pred]++
		}
		// Majority vote
		bestLabel := 0.0
		bestCount := 0
		for label, count := range votes {
			if count > bestCount {
				bestCount = count
				bestLabel = label
			}
		}
		preds[i] = bestLabel
	}
	return preds
}

// PredictProbability returns the probability of each class for all samples in x.
// Each row contains the fraction of trees voting for each class, ordered by class value.
func (rf *RandomForest) PredictProbability(x [][]float64) [][]float64 {
	nClasses := len(rf.classes)
	// Sort classes for consistent ordering
	sortedClasses := make([]float64, nClasses)
	copy(sortedClasses, rf.classes)
	sortFloat64s(sortedClasses)

	classIdx := make(map[float64]int)
	for i, c := range sortedClasses {
		classIdx[c] = i
	}

	probs := make([][]float64, len(x))
	nTrees := float64(rf.NTrees)

	for i, row := range x {
		votes := make([]float64, nClasses)
		for t, dt := range rf.trees {
			features := rf.featureIndices[t]
			subset := make([]float64, len(features))
			for j, f := range features {
				subset[j] = row[f]
			}
			pred := dt.Predict([][]float64{subset})[0]
			if idx, ok := classIdx[pred]; ok {
				votes[idx]++
			}
		}
		prob := make([]float64, nClasses)
		for j := range votes {
			prob[j] = votes[j] / nTrees
		}
		probs[i] = prob
	}
	return probs
}

// FeatureImportance returns the average feature importance across all trees,
// mapped back to the original feature space. Values are normalized to sum to 1.
func (rf *RandomForest) FeatureImportance() []float64 {
	importance := make([]float64, rf.nFeatures)
	for t, dt := range rf.trees {
		treeImp := dt.FeatureImportance()
		features := rf.featureIndices[t]
		for j, f := range features {
			if j < len(treeImp) {
				importance[f] += treeImp[j]
			}
		}
	}

	// Normalize
	total := 0.0
	for _, v := range importance {
		total += v
	}
	if total > 0 {
		for i := range importance {
			importance[i] /= total
		}
	}
	return importance
}

// OOBScore returns the out-of-bag accuracy score. Each sample is predicted only
// by trees that did not include it in their bootstrap sample.
func (rf *RandomForest) OOBScore(x [][]float64, y []float64) float64 {
	if len(rf.oobPredictions) == 0 {
		return 0
	}

	correct := 0
	total := 0
	for i, votes := range rf.oobPredictions {
		if i >= len(y) {
			continue
		}
		// Find majority vote from OOB predictions
		bestLabel := 0.0
		bestCount := 0
		for label, count := range votes {
			if count > bestCount {
				bestCount = count
				bestLabel = label
			}
		}
		if bestLabel == y[i] {
			correct++
		}
		total++
	}

	if total == 0 {
		return 0
	}
	return float64(correct) / float64(total)
}

// sortFloat64s sorts a slice of float64 in ascending order (insertion sort).
func sortFloat64s(a []float64) {
	for i := 1; i < len(a); i++ {
		key := a[i]
		j := i - 1
		for j >= 0 && a[j] > key {
			a[j+1] = a[j]
			j--
		}
		a[j+1] = key
	}
}

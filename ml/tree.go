package ml

import "math"

// DecisionTree implements a classification decision tree using CART
// (Classification and Regression Trees) with configurable split criteria.
type DecisionTree struct {
	MaxDepth    int
	MinSamples  int
	Criterion   string // "gini" or "entropy"
	root        *treeNode
	nFeatures   int
	importances []float64
}

// DecisionTreeConfig holds configuration for creating a DecisionTree.
type DecisionTreeConfig struct {
	MaxDepth   int
	MinSamples int
	Criterion  string
}

type treeNode struct {
	featureIdx int
	threshold  float64
	left       *treeNode
	right      *treeNode
	label      float64
	isLeaf     bool
	samples    int
	impurity   float64
}

// NewDecisionTree creates a new DecisionTree with the given configuration.
// Defaults: MaxDepth=10, MinSamples=2, Criterion="gini".
func NewDecisionTree(config DecisionTreeConfig) *DecisionTree {
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
	return &DecisionTree{
		MaxDepth:   maxDepth,
		MinSamples: minSamples,
		Criterion:  criterion,
	}
}

// Fit builds the decision tree from training features x and labels y.
func (dt *DecisionTree) Fit(x [][]float64, y []float64) {
	if len(x) == 0 {
		return
	}
	dt.nFeatures = len(x[0])
	dt.importances = make([]float64, dt.nFeatures)
	dt.root = dt.buildTree(x, y, 0)

	// Normalize importances
	total := 0.0
	for _, imp := range dt.importances {
		total += imp
	}
	if total > 0 {
		for i := range dt.importances {
			dt.importances[i] /= total
		}
	}
}

// Predict returns predicted class labels for all samples in x.
func (dt *DecisionTree) Predict(x [][]float64) []float64 {
	preds := make([]float64, len(x))
	for i, row := range x {
		preds[i] = dt.predictSingle(dt.root, row)
	}
	return preds
}

// FeatureImportance returns the normalized importance of each feature
// based on impurity reduction.
func (dt *DecisionTree) FeatureImportance() []float64 {
	result := make([]float64, len(dt.importances))
	copy(result, dt.importances)
	return result
}

func (dt *DecisionTree) buildTree(x [][]float64, y []float64, depth int) *treeNode {
	n := len(y)

	// Check stopping conditions
	if depth >= dt.MaxDepth || n < dt.MinSamples || allSame(y) {
		return &treeNode{
			isLeaf:  true,
			label:   majorityClass(y),
			samples: n,
		}
	}

	bestFeature := -1
	bestThreshold := 0.0
	bestGain := 0.0
	parentImpurity := dt.impurity(y)

	for f := 0; f < dt.nFeatures; f++ {
		thresholds := uniqueValues(x, f)
		for _, thresh := range thresholds {
			leftY, rightY := splitByThreshold(x, y, f, thresh)
			if len(leftY) == 0 || len(rightY) == 0 {
				continue
			}

			leftImpurity := dt.impurity(leftY)
			rightImpurity := dt.impurity(rightY)

			weightedImpurity := (float64(len(leftY))*leftImpurity +
				float64(len(rightY))*rightImpurity) / float64(n)
			gain := parentImpurity - weightedImpurity

			if gain > bestGain {
				bestGain = gain
				bestFeature = f
				bestThreshold = thresh
			}
		}
	}

	if bestFeature == -1 || bestGain <= 0 {
		return &treeNode{
			isLeaf:  true,
			label:   majorityClass(y),
			samples: n,
		}
	}

	// Record feature importance
	dt.importances[bestFeature] += bestGain * float64(n)

	leftX, leftY, rightX, rightY := splitData(x, y, bestFeature, bestThreshold)

	return &treeNode{
		featureIdx: bestFeature,
		threshold:  bestThreshold,
		left:       dt.buildTree(leftX, leftY, depth+1),
		right:      dt.buildTree(rightX, rightY, depth+1),
		samples:    n,
		impurity:   parentImpurity,
	}
}

func (dt *DecisionTree) predictSingle(node *treeNode, x []float64) float64 {
	if node.isLeaf {
		return node.label
	}
	if x[node.featureIdx] <= node.threshold {
		return dt.predictSingle(node.left, x)
	}
	return dt.predictSingle(node.right, x)
}

func (dt *DecisionTree) impurity(y []float64) float64 {
	if dt.Criterion == "entropy" {
		return entropy(y)
	}
	return gini(y)
}

func gini(y []float64) float64 {
	counts := classCounts(y)
	n := float64(len(y))
	sum := 0.0
	for _, count := range counts {
		p := float64(count) / n
		sum += p * p
	}
	return 1 - sum
}

func entropy(y []float64) float64 {
	counts := classCounts(y)
	n := float64(len(y))
	sum := 0.0
	for _, count := range counts {
		p := float64(count) / n
		if p > 0 {
			sum -= p * math.Log2(p)
		}
	}
	return sum
}

func classCounts(y []float64) map[float64]int {
	counts := make(map[float64]int)
	for _, v := range y {
		counts[v]++
	}
	return counts
}

func majorityClass(y []float64) float64 {
	counts := classCounts(y)
	bestLabel := 0.0
	bestCount := 0
	for label, count := range counts {
		if count > bestCount {
			bestCount = count
			bestLabel = label
		}
	}
	return bestLabel
}

func allSame(y []float64) bool {
	if len(y) == 0 {
		return true
	}
	first := y[0]
	for _, v := range y[1:] {
		if v != first {
			return false
		}
	}
	return true
}

func uniqueValues(x [][]float64, col int) []float64 {
	seen := make(map[float64]bool)
	var vals []float64
	for _, row := range x {
		v := row[col]
		if !seen[v] {
			seen[v] = true
			vals = append(vals, v)
		}
	}
	return vals
}

func splitByThreshold(x [][]float64, y []float64, feature int, threshold float64) (leftY, rightY []float64) {
	for i := range x {
		if x[i][feature] <= threshold {
			leftY = append(leftY, y[i])
		} else {
			rightY = append(rightY, y[i])
		}
	}
	return
}

func splitData(x [][]float64, y []float64, feature int, threshold float64) (leftX [][]float64, leftY []float64, rightX [][]float64, rightY []float64) {
	for i := range x {
		if x[i][feature] <= threshold {
			leftX = append(leftX, x[i])
			leftY = append(leftY, y[i])
		} else {
			rightX = append(rightX, x[i])
			rightY = append(rightY, y[i])
		}
	}
	return
}

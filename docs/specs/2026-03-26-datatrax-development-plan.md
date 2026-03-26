# Datatrax — Development Plan

## Vision

**Datatrax** is a Go toolkit for data engineers — type coercion, batch processing, pipeline utilities, and classic ML algorithms. Built for engineers who work in Go and need data processing + ML without switching to Python.

**Module:** `github.com/rbmuller/datatrax`
**Tagline:** "Data engineering and ML toolkit for Go"

---

## Phase 1 — Cleanup & Foundation (Week 1-2)

### 1.1 Repository Rename & Setup
- [x] Rename GitHub repo from `devtools` to `datatrax`
- [x] Update `go.mod` module path to `github.com/rbmuller/datatrax`
- [x] Update all internal imports
- [x] Add MIT license
- [x] Add `.github/workflows/ci.yml` (lint + test)
- [x] Add `.gitignore`

### 1.2 Package Restructure
```
datatrax/
├── coerce/         # Type coercion (Floatify, Integerify, Stringify, etc.)
├── batch/          # Batch processing (ChunkArray, pipeline utilities)
├── dedup/          # Deduplication (single generic implementation)
├── dateutil/       # Date/time utilities (EpochToTimestamp, DaysDifference, etc.)
├── strutil/        # String utilities (TrimQuotes, StringInArray, etc.)
├── maputil/        # Map operations (GenerateMap, CopyMap, etc.)
├── errutil/        # Enhanced error handling with file:line
├── mathutil/       # Safe division, basic math operations
├── ml/             # Classic ML algorithms (Phase 2)
├── examples/       # Usage examples
├── README.md
├── go.mod
└── go.sum
```

### 1.3 Code Quality
- [x] Remove `mail/` package entirely
- [x] Remove `logging/` package (replace with recommendation to use `slog`)
- [x] Consolidate 3 dedup functions into 1 generic `Deduplicate[T comparable]`
- [x] Consolidate search functions into generic `Contains[T comparable]`
- [x] Standardize all return types to `(value, error)` pattern
- [x] Replace `models.Maps` with `map[string]any`
- [x] Replace `DatePair` strings with `time.Time`
- [x] Add godoc comments to ALL public functions
- [x] Add unit tests for ALL packages (47 tests, all passing)
- [x] Update dependencies to latest versions
- [x] Remove `gopkg.in/guregu/null.v3` — zero external dependencies now

---

## Phase 2 — Classic ML Package (Week 3-6)

### 2.1 Core ML Infrastructure
```
ml/
├── dataset.go        # Dataset loading, splitting, normalization
├── metrics.go        # Accuracy, precision, recall, F1, MSE, R²
├── preprocessing.go  # Feature scaling, encoding, missing values
├── linear.go         # Linear Regression
├── logistic.go       # Logistic Regression
├── knn.go            # K-Nearest Neighbors
├── kmeans.go         # K-Means Clustering
├── tree.go           # Decision Tree (classification + regression)
├── naivebayes.go     # Naive Bayes Classifier
├── ml_test.go        # Tests for all algorithms
└── examples/         # Usage examples with real datasets
```

### 2.2 Algorithms — Priority Order

#### Linear Regression
- [ ] Simple linear regression (single feature)
- [ ] Multiple linear regression (multi-feature)
- [ ] Gradient descent optimizer
- [ ] Closed-form solution (normal equation)
- [ ] R², MSE, MAE metrics
- [ ] Predict function

#### Logistic Regression
- [ ] Binary classification
- [ ] Sigmoid function
- [ ] Gradient descent with learning rate
- [ ] Accuracy, precision, recall, F1
- [ ] Predict + PredictProbability

#### K-Nearest Neighbors (KNN)
- [ ] Classification and regression
- [ ] Euclidean, Manhattan distance metrics
- [ ] Configurable K parameter
- [ ] Weighted voting option

#### K-Means Clustering
- [ ] Lloyd's algorithm
- [ ] K-Means++ initialization
- [ ] Configurable K and max iterations
- [ ] Inertia / silhouette score
- [ ] Predict cluster assignment

#### Decision Tree
- [ ] Classification (Gini impurity, entropy)
- [ ] Regression (variance reduction)
- [ ] Configurable max depth, min samples
- [ ] Feature importance
- [ ] Tree visualization (text-based)

#### Naive Bayes
- [ ] Gaussian Naive Bayes
- [ ] Multinomial Naive Bayes
- [ ] Predict + PredictProbability

### 2.3 Shared ML Infrastructure
- [ ] `Dataset` struct: Load from CSV, slice, shuffle, split (train/test)
- [ ] `Scaler`: MinMax, StandardScaler (z-score)
- [ ] `Encoder`: OneHot, Label encoding
- [ ] `Metrics`: Accuracy, Precision, Recall, F1, ConfusionMatrix, MSE, RMSE, R²
- [ ] `CrossValidation`: K-Fold cross validation

### 2.4 API Design
```go
// Clean, consistent API across all algorithms
model := ml.NewLinearRegression()
model.Fit(xTrain, yTrain)
predictions := model.Predict(xTest)
score := ml.R2Score(yTest, predictions)

// Same pattern for all models
clf := ml.NewKNN(ml.KNNConfig{K: 5, Distance: ml.Euclidean})
clf.Fit(xTrain, yTrain)
predictions := clf.Predict(xTest)
accuracy := ml.Accuracy(yTest, predictions)

// Clustering
km := ml.NewKMeans(ml.KMeansConfig{K: 3, MaxIter: 100})
km.Fit(data)
labels := km.Predict(newData)
```

---

## Phase 3 — README & Documentation (Week 7)

### 3.1 README Structure
- [ ] Hero banner (name + tagline)
- [ ] Badges (CI, coverage, Go version, license, Go Report Card)
- [ ] Install: `go get github.com/rbmuller/datatrax`
- [ ] Quick start (3 examples: coerce, batch, ML)
- [ ] Package overview table
- [ ] ML algorithms comparison table (accuracy, speed, use case)
- [ ] Benchmarks vs Python scikit-learn (inference speed)
- [ ] Contributing guide
- [ ] License

### 3.2 Examples
- [ ] `examples/pipeline/` — batch processing + coercion pipeline
- [ ] `examples/classification/` — KNN on Iris dataset
- [ ] `examples/regression/` — Linear regression on housing data
- [ ] `examples/clustering/` — K-Means on synthetic data

---

## Phase 4 — Divulgação (Week 8+)

### 4.1 Launch Checklist
- [ ] Post no LinkedIn (English): "I built a Go ML toolkit because Python isn't always the answer"
- [ ] Post no Reddit r/golang: Show benchmarks vs scikit-learn
- [ ] Submit PR to [awesome-go](https://github.com/avelino/awesome-go) under Machine Learning
- [ ] Post no Hacker News (Show HN)
- [ ] Tweet/post com benchmark comparisons

### 4.2 Ongoing
- [ ] Responder issues rapidamente
- [ ] Aceitar PRs da comunidade
- [ ] Adicionar novos algoritmos baseado em requests (Random Forest, SVM, PCA)
- [ ] Blog post: "Classic ML in Go — When Python is Overkill"

---

## Success Metrics

| Milestone | Target | Timeline |
|-----------|--------|----------|
| First release (v0.1.0) | Cleanup complete, tests passing | Week 2 |
| ML release (v0.5.0) | 6 algorithms implemented | Week 6 |
| Public launch | README, examples, benchmarks | Week 7 |
| 100 stars | Divulgação + Awesome Go | Week 10 |
| 500 stars | Community + word of mouth | Month 6 |
| First external PR | Community engagement | Month 3 |

---

## Differentiators vs Existing Go ML Libs

| Library | Status | Why Datatrax is different |
|---------|--------|--------------------------|
| goml | Abandoned (2019) | Active, modern Go 1.21+ |
| golearn | Abandoned (2020) | Simpler API, batteries included |
| gorgonia | Active but complex | Datatrax is scikit-learn-simple, not TensorFlow-complex |
| sajari/regression | Only regression | Full toolkit: coercion + batch + ML |

**Positioning:** Datatrax is NOT competing with deep learning frameworks. It's the **scikit-learn of Go** — classic ML algorithms with a clean API, plus data engineering utilities that no other Go ML lib offers.

---

## Tech Stack

- **Go 1.21+** (generics, slog)
- **Zero external dependencies for ML** (pure Go, stdlib only)
- **gonum** only if needed for matrix operations
- **GitHub Actions** for CI/CD
- **Codecov** for coverage tracking

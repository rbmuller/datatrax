<div align="center">

# Datatrax

**Data engineering and machine learning toolkit for Go.**

Batch processing, type coercion, deduplication, date utilities, and classic ML algorithms — all in pure Go with zero external dependencies.

[![Go Reference](https://pkg.go.dev/badge/github.com/rbmuller/datatrax.svg)](https://pkg.go.dev/github.com/rbmuller/datatrax)
[![CI](https://github.com/rbmuller/datatrax/actions/workflows/ci.yml/badge.svg)](https://github.com/rbmuller/datatrax/actions/workflows/ci.yml)
[![Go Report Card](https://goreportcard.com/badge/github.com/rbmuller/datatrax)](https://goreportcard.com/report/github.com/rbmuller/datatrax)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Go Version](https://img.shields.io/github/go-mod/go-version/rbmuller/datatrax)](go.mod)

</div>

---

## Why Datatrax?

Most data engineers use **Go for pipelines** and **Python for everything else**. Datatrax eliminates the context switch — type coercion, batch processing, deduplication, and classic ML all in one Go module.

- **Zero dependencies** — pure Go stdlib, nothing to audit
- **Generics-first** — built for Go 1.21+, type-safe by default
- **Battle-tested utilities** — born from real-world ETL pipelines processing 500k+ records/day
- **ML without Python** — classic algorithms with a scikit-learn-simple API (coming soon)

## Install

```bash
go get github.com/rbmuller/datatrax
```

## Packages

| Package | Description | Key Functions |
|---------|-------------|---------------|
| [`batch`](batch/) | Split slices into chunks for parallel processing | `ChunkArray[T]` |
| [`coerce`](coerce/) | Convert `interface{}` to typed values safely | `Floatify`, `Integerify`, `Boolify`, `Stringify` |
| [`dateutil`](dateutil/) | Date/time parsing, conversion, and math | `EpochToTimestamp`, `DaysDifference`, `StringToDate` |
| [`dedup`](dedup/) | Remove duplicates from any comparable slice | `Deduplicate[T]` |
| [`errutil`](errutil/) | Errors with automatic file:line location | `NewError` |
| [`maputil`](maputil/) | Map operations — copy, generate from JSON | `CopyMap[K,V]`, `GenerateMap` |
| [`mathutil`](mathutil/) | Safe math operations | `Divide` (zero-safe) |
| [`strutil`](strutil/) | String utilities and generic search | `Contains[T]`, `TrimQuotes`, `SplitByRegexp` |
| [`ml`](ml/) | Classic ML algorithms — 6 models, metrics, preprocessing | `LinearRegression`, `KNN`, `KMeans`, `DecisionTree`, ... |

## Quick Start

### Batch Processing

Split large datasets into manageable chunks for parallel processing:

```go
import "github.com/rbmuller/datatrax/batch"

records := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
chunks := batch.ChunkArray(records, 3)
// [[1 2 3] [4 5 6] [7 8 9] [10]]

// Process chunks in parallel
for _, chunk := range chunks {
    go processChunk(chunk)
}
```

### Type Coercion

Safely convert untyped data from JSON, CSV, or database results:

```go
import "github.com/rbmuller/datatrax/coerce"

val, err := coerce.Floatify("3.14")    // 3.14, nil
val, err := coerce.Floatify(42)        // 42.0, nil
val, err := coerce.Integerify("100")   // 100, nil
val, err := coerce.Boolify(1)          // true, nil
val, err := coerce.Stringify(3.14)     // "3.14", nil
```

### Deduplication

Remove duplicates from any comparable slice — strings, ints, structs:

```go
import "github.com/rbmuller/datatrax/dedup"

names := []string{"Alice", "Bob", "Alice", "Charlie", "Bob"}
unique := dedup.Deduplicate(names)
// ["Alice", "Bob", "Charlie"]

ids := []int{1, 2, 3, 2, 1, 4}
unique := dedup.Deduplicate(ids)
// [1, 2, 3, 4]
```

### Date Utilities

Parse, convert, and calculate date differences:

```go
import "github.com/rbmuller/datatrax/dateutil"

// Convert epoch milliseconds to readable timestamp
ts, ok := dateutil.EpochToTimestamp(1684624830053)
// "2023-05-21 02:00:30"

// Calculate days between dates
days, err := dateutil.DaysDifference("2024-01-01", "2024-03-15", "2006-01-02")
// 74

// Parse date strings
t, err := dateutil.StringToDate("2024-03-15", "2006-01-02")
```

### Error Utilities

Wrap errors with automatic source file and line number:

```go
import "github.com/rbmuller/datatrax/errutil"

err := errutil.NewError(errors.New("connection timeout"))
fmt.Println(err)
// "main.go:42 - connection timeout"

// Supports errors.Is / errors.As via Unwrap()
errors.Is(err, originalErr) // true
```

### String Utilities

Generic search, trimming, and formatting:

```go
import "github.com/rbmuller/datatrax/strutil"

// Generic contains — works with any comparable type
strutil.Contains([]string{"a", "b", "c"}, "b")  // true
strutil.Contains([]int{1, 2, 3}, 5)              // false

// Trim surrounding quotes
strutil.TrimQuotes(`"hello world"`)  // "hello world"

// Join with quotes for SQL
strutil.StringifyWithQuotes([]string{"a", "b"})  // "'a','b'"

// Safe index access — no panics
strutil.SafeIndex([]string{"a", "b"}, 5)  // "", false
```

### Map Utilities

Copy maps and parse JSON:

```go
import "github.com/rbmuller/datatrax/maputil"

// Generic shallow copy
original := map[string]int{"a": 1, "b": 2}
copied := maputil.CopyMap(original)

// Parse JSON bytes to map
data := []byte(`{"name": "datatrax", "version": 1}`)
m, err := maputil.GenerateMap(data)
```

### Safe Math

Division without panics:

```go
import "github.com/rbmuller/datatrax/mathutil"

mathutil.Divide(10, 3)  // 3.333...
mathutil.Divide(10, 0)  // 0 (no panic)
```

## Machine Learning

6 classic ML algorithms with a consistent `Fit` / `Predict` API — pure Go, zero dependencies.

| Algorithm | Type | Key Config |
|-----------|------|------------|
| `LinearRegression` | Regression | LearningRate, Epochs (+ Normal Equation) |
| `LogisticRegression` | Classification | LearningRate, Epochs, Threshold |
| `KNN` | Classification | K, Distance (euclidean/manhattan) |
| `KMeans` | Clustering | K, MaxIter (K-Means++ init) |
| `DecisionTree` | Classification | MaxDepth, MinSamples, Criterion (gini/entropy) |
| `GaussianNB` | Classification | — (parameter-free) |

**Infrastructure:** Dataset (CSV loading, train/test split), Preprocessing (MinMaxScale, StandardScale), Metrics (Accuracy, Precision, Recall, F1, MSE, RMSE, MAE, R²), K-Fold Cross Validation.

### Linear Regression

```go
import "github.com/rbmuller/datatrax/ml"

model := ml.NewLinearRegression()
model.Fit(xTrain, yTrain)
predictions := model.Predict(xTest)
fmt.Println("R²:", ml.R2Score(yTest, predictions))
```

### Classification (KNN)

```go
clf := ml.NewKNN(ml.KNNConfig{K: 5, Distance: "euclidean"})
clf.Fit(xTrain, yTrain)
predictions := clf.Predict(xTest)
fmt.Println("Accuracy:", ml.Accuracy(yTest, predictions))
fmt.Println("F1:", ml.F1Score(yTest, predictions, 1.0))
```

### Clustering (K-Means)

```go
km := ml.NewKMeans(ml.KMeansConfig{K: 3, MaxIter: 100})
km.Fit(data)
labels := km.Predict(data)
fmt.Println("Inertia:", km.Inertia())
```

### Decision Tree

```go
dt := ml.NewDecisionTree(ml.DecisionTreeConfig{
    MaxDepth:   5,
    MinSamples: 2,
    Criterion:  "gini",
})
dt.Fit(xTrain, yTrain)
predictions := dt.Predict(xTest)
fmt.Println("Importance:", dt.FeatureImportance())
```

### Preprocessing & Evaluation

```go
// Scale features
xScaled := ml.MinMaxScale(xTrain)

// Cross validation
folds := ml.KFoldSplit(x, y, 5)
for _, fold := range folds {
    model.Fit(fold.XTrain, fold.YTrain)
    pred := model.Predict(fold.XTest)
    fmt.Println("Fold R²:", ml.R2Score(fold.YTest, pred))
}

// Full metrics
fmt.Println("Accuracy:", ml.Accuracy(yTrue, yPred))
fmt.Println("Precision:", ml.Precision(yTrue, yPred, 1.0))
fmt.Println("Recall:", ml.Recall(yTrue, yPred, 1.0))
fmt.Println("Confusion:", ml.ConfusionMatrix(yTrue, yPred))
```

### Load Dataset from CSV

```go
dataset, err := ml.LoadCSV("data.csv", 4) // target is column 4
xTrain, xTest, yTrain, yTest := dataset.Split(0.8)
```

## Roadmap

| Version | What | Status |
|---------|------|--------|
| **v0.1.0** | Core utilities — 8 packages, 47 tests, zero deps | **Done** |
| **v0.5.0** | Classic ML — 6 algorithms, preprocessing, metrics, cross-validation | **Done** |
| **v1.0.0** | Benchmarks vs scikit-learn, tree visualization, encoders, multinomial NB | Planned |

## Design Principles

1. **Zero dependencies** — If it can be done with stdlib, it will be
2. **Generics everywhere** — Type safety is not optional
3. **No silent failures** — Functions return `(value, error)`, not zero values
4. **Pipeline-ready** — Every function works with slices and streams
5. **Documentation-driven** — If it's not documented, it doesn't exist

## Contributing

Contributions are welcome! Please:

1. Fork the repo
2. Create a feature branch (`git checkout -b feat/amazing-feature`)
3. Write tests for your changes
4. Ensure `go test -race ./...` passes
5. Open a PR

## License

[MIT](LICENSE) — Robson Muller, 2026

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
| `ml` | Classic ML algorithms *(coming in v0.5.0)* | Linear Regression, KNN, K-Means, Decision Tree, ... |

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

## Roadmap

| Version | What | Status |
|---------|------|--------|
| **v0.1.0** | Core utilities — 8 packages, 47 tests, zero deps | **Done** |
| **v0.5.0** | Classic ML — Linear/Logistic Regression, KNN, K-Means, Decision Tree, Naive Bayes | In Progress |
| **v1.0.0** | Full ML suite — preprocessing, cross-validation, metrics, benchmarks | Planned |

### ML Preview (v0.5.0)

```go
import "github.com/rbmuller/datatrax/ml"

// Train a model in 3 lines
model := ml.NewLinearRegression()
model.Fit(xTrain, yTrain)
predictions := model.Predict(xTest)

// Evaluate
score := ml.R2Score(yTest, predictions)

// Same clean API for all algorithms
clf := ml.NewKNN(ml.KNNConfig{K: 5})
clf.Fit(xTrain, yTrain)
accuracy := ml.Accuracy(yTest, clf.Predict(xTest))
```

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

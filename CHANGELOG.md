# Changelog

All notable changes to Datatrax are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.2.0] — 2026-03-31

### Added
- **Random Forest classifier** (`ml/forest.go`) — bootstrap sampling, feature bagging (`sqrt(n_features)` by default), out-of-bag (OOB) score for free validation, feature importance averaged across trees, and probability prediction.
- **`CONTRIBUTING.md`** — contribution guide covering local setup, test/benchmark commands, project structure, and the "what we're looking for" list (SVM, PCA, ensemble methods, perf, docs).

### Changed
- README ML table and roadmap row updated to reflect Random Forest shipping; v2.0.0 row now lists SVM, PCA, ensemble methods.

## [1.1.0] — 2026-03-26

### Added
- **`ml` package** — full classic-ML toolkit, pure Go, zero external dependencies.
  - **Algorithms (7):** Linear Regression (gradient descent + normal equation), Logistic Regression, KNN (euclidean/manhattan + optional inverse-distance weighting), K-Means (Lloyd's algorithm + K-Means++ init), Decision Tree (gini/entropy, `String()` text viz, feature importance), Gaussian Naive Bayes, Multinomial Naive Bayes (Laplace smoothing).
  - **Dataset infra:** `LoadCSV`, `Split` (train/test), `Shuffle`, `Shape`.
  - **Preprocessing:** `MinMaxScale`, `StandardScale`, column-wise statistics.
  - **Encoding:** `LabelEncode` / `LabelDecode`, `OneHotEncode`.
  - **Metrics:** `Accuracy`, `Precision`, `Recall`, `F1Score`, `ConfusionMatrix`, `MSE`, `RMSE`, `MAE`, `R2Score`.
  - **Cross-validation:** `KFoldSplit`.
  - **Benchmarks:** `ml/benchmark_test.go` (Apple M4 reference numbers in README).
  - **Examples:** `examples/classification/`, `examples/regression/`, `examples/clustering/`.
- **CI workflow** (`.github/workflows/ci.yml`) — `go vet` + `go test -race -coverprofile`.
- **MIT license** and Go Report Card A+ tracking.
- Listed in [awesome-go](https://github.com/avelino/awesome-go) under Machine Learning (PR #6161).

### Changed
- **Repository renamed** from `devtools` to `datatrax`. Module path is now `github.com/rbmuller/datatrax`.
- **Package layout restructured** into 8 focused packages: `batch`, `coerce`, `dateutil`, `dedup`, `errutil`, `maputil`, `mathutil`, `strutil`.
- **API consolidated:** three legacy dedup functions collapsed into one generic `Deduplicate[T comparable]`; legacy search helpers collapsed into `Contains[T comparable]`. All public functions return `(value, error)` where applicable.
- **Type modernization:** `models.Maps` replaced by `map[string]any`; `DatePair` strings replaced by `time.Time`.
- All public functions now have godoc comments. 47 unit tests, all passing.

### Removed
- `gopkg.in/guregu/null.v3` dependency dropped — repository is now zero-dep stdlib-only.
- Legacy `mail/` and `logging/` packages removed (`slog` is the recommended replacement for the latter).

## [1.0.0] — 2026-03-26

Same commit as `v1.1.0`. Tagged for compatibility with consumers pinning to a 1.0 line; new work should track `v1.1.0` and later.

## [1.0.1] — 2023-11-23

Final release under the old `devtools` name and module path. Pre-dates the toolkit rename and the `ml` package. Kept here for historical reference; not recommended for new code — pin `v1.1.0` or later (`github.com/rbmuller/datatrax`).

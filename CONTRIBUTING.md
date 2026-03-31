# Contributing to Datatrax

Thanks for your interest in contributing! Here's how to get started.

## Quick Start

```bash
git clone https://github.com/rbmuller/datatrax.git
cd datatrax
go test ./...
```

## How to Contribute

1. **Fork** the repo
2. **Create a branch** (`git checkout -b feat/my-feature`)
3. **Write tests** for your changes
4. **Run checks** before committing:
   ```bash
   go vet ./...
   go test -race ./...
   ```
5. **Open a PR** with a clear description

## What We're Looking For

- New ML algorithms (SVM, PCA, ensemble methods)
- Performance optimizations with benchmarks
- Bug fixes with test cases
- Documentation improvements

## Guidelines

- **Zero external dependencies** — stdlib only
- **Generics-first** — use Go 1.21+ generics where applicable
- **Tests required** — target 90%+ coverage for new code
- **Godoc comments** on all public functions
- **Consistent API** — ML models must implement `Fit()` and `Predict()`
- **No breaking changes** without discussion first

## Running Benchmarks

```bash
go test -bench=. -benchmem ./...
```

## Project Structure

```
datatrax/
├── batch/       # Batch processing
├── coerce/      # Type coercion
├── dateutil/    # Date utilities
├── dedup/       # Deduplication
├── errutil/     # Error utilities
├── maputil/     # Map operations
├── mathutil/    # Math utilities
├── strutil/     # String utilities
├── ml/          # Machine learning
└── examples/    # Usage examples
```

## Questions?

Open an issue or reach out to [@rbmuller](https://github.com/rbmuller).

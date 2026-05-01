---
name: New ML algorithm
about: Propose a new algorithm to add to the ml package
title: 'algo: '
labels: enhancement
---

## Algorithm

<!-- Name and category (classification / regression / clustering / dimensionality reduction / preprocessing) -->

## Why this fits Datatrax

<!--
Datatrax is the "scikit-learn of Go" — classic ML, pure stdlib, zero external deps.
Algorithms that fit:
  - Documented, well-understood, with reference papers or textbook treatment
  - Implementable in pure Go without numerical libs (or with vanilla matrix code)
  - Useful on the kind of tabular data engineers actually have
Things that don't fit:
  - Deep learning (use gorgonia)
  - Heavy GPU-bound algorithms
  - Anything requiring CGo or external C libraries
-->

## API sketch

```go
// Match the existing Fit/Predict pattern. Example:
m := ml.NewYourAlgo(ml.YourAlgoConfig{...})
m.Fit(xTrain, yTrain)
preds := m.Predict(xTest)
```

## Implementation notes

<!-- Reference papers, complexity, edge cases, hyperparameter defaults -->

## Acceptance

- [ ] Pure Go, zero external dependencies
- [ ] Godoc on all exported symbols
- [ ] Tests with deterministic seed (target ≥90% line coverage for the new file)
- [ ] Benchmark added to `ml/benchmark_test.go` (Fit + Predict)
- [ ] README ML table updated

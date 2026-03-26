// Package batch provides utilities for splitting slices into fixed-size chunks
// for batch processing workloads.
package batch

// ChunkArray splits a slice into sub-slices of the given chunk size.
// The last chunk may contain fewer elements than chunkSize.
// It panics if chunkSize is less than 1.
func ChunkArray[T any](items []T, chunkSize int) [][]T {
	if chunkSize < 1 {
		panic("batch: chunkSize must be >= 1")
	}

	chunks := make([][]T, 0, (len(items)+chunkSize-1)/chunkSize)
	for chunkSize < len(items) {
		items, chunks = items[chunkSize:], append(chunks, items[0:chunkSize:chunkSize])
	}

	return append(chunks, items)
}

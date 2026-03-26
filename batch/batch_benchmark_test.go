package batch

import "testing"

func BenchmarkChunkArray(b *testing.B) {
	items := make([]int, 10000)
	for i := range items {
		items[i] = i
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ChunkArray(items, 100)
	}
}

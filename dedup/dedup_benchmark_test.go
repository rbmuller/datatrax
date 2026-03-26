package dedup

import (
	"fmt"
	"testing"
)

func BenchmarkDeduplicate(b *testing.B) {
	items := make([]string, 10000)
	for i := range items {
		// Use mod 5000 so roughly 50% are duplicates
		items[i] = fmt.Sprintf("item-%d", i%5000)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Deduplicate(items)
	}
}

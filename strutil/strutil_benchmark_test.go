package strutil

import (
	"fmt"
	"testing"
)

func BenchmarkContains(b *testing.B) {
	items := make([]string, 10000)
	for i := range items {
		items[i] = fmt.Sprintf("element-%d", i)
	}
	// Search for an element near the end to exercise the full scan
	target := "element-9999"

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Contains(items, target)
	}
}

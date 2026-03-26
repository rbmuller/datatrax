package batch

import (
	"testing"
)

func TestChunkArray(t *testing.T) {
	tests := []struct {
		name      string
		items     []int
		chunkSize int
		wantLen   int
	}{
		{"even split", []int{1, 2, 3, 4}, 2, 2},
		{"uneven split", []int{1, 2, 3, 4, 5}, 2, 3},
		{"single chunk", []int{1, 2, 3}, 5, 1},
		{"chunk size 1", []int{1, 2, 3}, 1, 3},
		{"empty slice", []int{}, 3, 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ChunkArray(tt.items, tt.chunkSize)
			if len(got) != tt.wantLen {
				t.Errorf("ChunkArray() returned %d chunks, want %d", len(got), tt.wantLen)
			}
		})
	}
}

func TestChunkArrayPanicsOnZero(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("ChunkArray() did not panic on chunkSize 0")
		}
	}()
	ChunkArray([]int{1, 2, 3}, 0)
}

func TestChunkArrayPreservesElements(t *testing.T) {
	items := []int{1, 2, 3, 4, 5, 6, 7}
	chunks := ChunkArray(items, 3)

	total := 0
	for _, chunk := range chunks {
		total += len(chunk)
	}
	if total != len(items) {
		t.Errorf("total elements = %d, want %d", total, len(items))
	}
}

func TestChunkArrayStrings(t *testing.T) {
	items := []string{"a", "b", "c", "d"}
	chunks := ChunkArray(items, 2)
	if len(chunks) != 2 {
		t.Errorf("expected 2 chunks, got %d", len(chunks))
	}
	if chunks[0][0] != "a" || chunks[0][1] != "b" {
		t.Errorf("first chunk = %v, want [a b]", chunks[0])
	}
}

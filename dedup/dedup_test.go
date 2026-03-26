package dedup

import (
	"testing"
)

func TestDeduplicateStrings(t *testing.T) {
	input := []string{"a", "b", "a", "c", "b"}
	got := Deduplicate(input)
	want := []string{"a", "b", "c"}
	if len(got) != len(want) {
		t.Fatalf("Deduplicate() len = %d, want %d", len(got), len(want))
	}
	for i, v := range got {
		if v != want[i] {
			t.Errorf("Deduplicate()[%d] = %q, want %q", i, v, want[i])
		}
	}
}

func TestDeduplicateInts(t *testing.T) {
	input := []int{1, 2, 3, 2, 1, 4}
	got := Deduplicate(input)
	want := []int{1, 2, 3, 4}
	if len(got) != len(want) {
		t.Fatalf("Deduplicate() len = %d, want %d", len(got), len(want))
	}
	for i, v := range got {
		if v != want[i] {
			t.Errorf("Deduplicate()[%d] = %d, want %d", i, v, want[i])
		}
	}
}

func TestDeduplicateEmpty(t *testing.T) {
	got := Deduplicate([]string{})
	if len(got) != 0 {
		t.Errorf("Deduplicate(empty) len = %d, want 0", len(got))
	}
}

func TestDeduplicateNoDuplicates(t *testing.T) {
	input := []string{"x", "y", "z"}
	got := Deduplicate(input)
	if len(got) != 3 {
		t.Errorf("Deduplicate() len = %d, want 3", len(got))
	}
}

func TestDeduplicateAllSame(t *testing.T) {
	input := []int{7, 7, 7, 7}
	got := Deduplicate(input)
	if len(got) != 1 || got[0] != 7 {
		t.Errorf("Deduplicate() = %v, want [7]", got)
	}
}

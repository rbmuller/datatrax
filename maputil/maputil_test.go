package maputil

import (
	"testing"
)

func TestGenerateMap(t *testing.T) {
	data := []byte(`{"key": "value", "num": 42}`)
	got, err := GenerateMap(data)
	if err != nil {
		t.Fatalf("GenerateMap() error = %v", err)
	}
	if got["key"] != "value" {
		t.Errorf("GenerateMap() key = %v, want \"value\"", got["key"])
	}
}

func TestGenerateMapInvalidJSON(t *testing.T) {
	_, err := GenerateMap([]byte("not json"))
	if err == nil {
		t.Error("GenerateMap() should return error for invalid JSON")
	}
}

func TestGenerateMapEmpty(t *testing.T) {
	got, err := GenerateMap([]byte("{}"))
	if err != nil {
		t.Fatalf("GenerateMap() error = %v", err)
	}
	if len(got) != 0 {
		t.Errorf("GenerateMap({}) len = %d, want 0", len(got))
	}
}

func TestCopyMap(t *testing.T) {
	original := map[string]int{"a": 1, "b": 2}
	copied := CopyMap(original)

	if len(copied) != len(original) {
		t.Errorf("CopyMap() len = %d, want %d", len(copied), len(original))
	}
	if copied["a"] != 1 || copied["b"] != 2 {
		t.Errorf("CopyMap() values don't match")
	}

	// Verify it's a separate copy
	copied["c"] = 3
	if _, ok := original["c"]; ok {
		t.Error("CopyMap() should create independent copy")
	}
}

func TestCopyMapEmpty(t *testing.T) {
	got := CopyMap(map[string]int{})
	if len(got) != 0 {
		t.Errorf("CopyMap(empty) len = %d, want 0", len(got))
	}
}

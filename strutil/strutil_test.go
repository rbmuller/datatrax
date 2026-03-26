package strutil

import (
	"testing"
)

func TestTrimQuotes(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{`"hello"`, "hello"},
		{`  "world"  `, "world"},
		{"no quotes", "no quotes"},
		{`"a"`, "a"},
		{`""`, ""},
	}
	for _, tt := range tests {
		got := TrimQuotes(tt.input)
		if got != tt.want {
			t.Errorf("TrimQuotes(%q) = %q, want %q", tt.input, got, tt.want)
		}
	}
}

func TestMapArrayToString(t *testing.T) {
	tests := []struct {
		input []string
		want  string
	}{
		{[]string{"a", "b", "c"}, "a,b,c"},
		{[]string{"x"}, "x"},
		{[]string{}, ""},
	}
	for _, tt := range tests {
		got := MapArrayToString(tt.input)
		if got != tt.want {
			t.Errorf("MapArrayToString(%v) = %q, want %q", tt.input, got, tt.want)
		}
	}
}

func TestNumbersToString(t *testing.T) {
	got := NumbersToString([]uint{1, 2, 3}, ",")
	if got != "1,2,3" {
		t.Errorf("NumbersToString() = %q, want \"1,2,3\"", got)
	}
}

func TestStringifyWithQuotes(t *testing.T) {
	tests := []struct {
		input []string
		want  string
	}{
		{[]string{"a", "b"}, "'a','b'"},
		{[]string{}, ""},
		{[]string{"x"}, "'x'"},
	}
	for _, tt := range tests {
		got := StringifyWithQuotes(tt.input)
		if got != tt.want {
			t.Errorf("StringifyWithQuotes(%v) = %q, want %q", tt.input, got, tt.want)
		}
	}
}

func TestSplitByRegexp(t *testing.T) {
	got := SplitByRegexp("a1b2c", "[0-9]")
	if len(got) != 3 || got[0] != "a" || got[1] != "b" || got[2] != "c" {
		t.Errorf("SplitByRegexp() = %v, want [a b c]", got)
	}
}

func TestContains(t *testing.T) {
	if !Contains([]string{"a", "b", "c"}, "b") {
		t.Error("Contains() should find 'b'")
	}
	if Contains([]string{"a", "b"}, "z") {
		t.Error("Contains() should not find 'z'")
	}
	if !Contains([]int{1, 2, 3}, 3) {
		t.Error("Contains() should find 3")
	}
}

func TestSafeIndex(t *testing.T) {
	list := []string{"a", "b", "c"}
	if SafeIndex(list, 1) != "b" {
		t.Error("SafeIndex(1) should return 'b'")
	}
	if SafeIndex(list, 10) != "" {
		t.Error("SafeIndex(10) should return zero value")
	}
	if SafeIndex(list, -1) != "" {
		t.Error("SafeIndex(-1) should return zero value")
	}
}

func TestAppendIfMissing(t *testing.T) {
	s := []string{"a", "b"}
	got := AppendIfMissing(s, "c")
	if len(got) != 3 {
		t.Errorf("AppendIfMissing() len = %d, want 3", len(got))
	}
	got = AppendIfMissing(got, "a")
	if len(got) != 3 {
		t.Errorf("AppendIfMissing() should not add duplicate, len = %d", len(got))
	}
}

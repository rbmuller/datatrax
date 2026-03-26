package coerce

import (
	"testing"
)

func TestFloatify(t *testing.T) {
	tests := []struct {
		name    string
		input   interface{}
		want    float64
		wantErr bool
	}{
		{"float64 input", 3.14, 3.14, false},
		{"float32 input", float32(1.5), 1.5, false},
		{"int input", 42, 42.0, false},
		{"int64 input", int64(99), 99.0, false},
		{"string input", "2.5", 2.5, false},
		{"invalid string", "abc", 0, true},
		{"nil input", nil, 0, true},
		{"bool input", true, 0, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := Floatify(tt.input)
			if (err != nil) != tt.wantErr {
				t.Errorf("Floatify() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("Floatify() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestIntegerify(t *testing.T) {
	tests := []struct {
		name    string
		input   interface{}
		want    int64
		wantErr bool
	}{
		{"float64 input", 3.9, 3, false},
		{"float32 input", float32(2.7), 2, false},
		{"int input", 42, 42, false},
		{"int64 input", int64(77), 77, false},
		{"string input", "100", 100, false},
		{"invalid string", "abc", 0, true},
		{"nil input", nil, 0, true},
		{"bool input (unsupported)", true, 0, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := Integerify(tt.input)
			if (err != nil) != tt.wantErr {
				t.Errorf("Integerify() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("Integerify() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestBoolify(t *testing.T) {
	tests := []struct {
		name    string
		input   interface{}
		want    bool
		wantErr bool
	}{
		{"true", true, true, false},
		{"false", false, false, false},
		{"string input", "true", false, true},
		{"nil input", nil, false, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := Boolify(tt.input)
			if (err != nil) != tt.wantErr {
				t.Errorf("Boolify() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("Boolify() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestStringify(t *testing.T) {
	tests := []struct {
		name    string
		input   interface{}
		want    string
		wantErr bool
	}{
		{"string input", "hello", "hello", false},
		{"empty string", "", "", false},
		{"nil input", nil, "", true},
		{"int input", 42, "", true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := Stringify(tt.input)
			if (err != nil) != tt.wantErr {
				t.Errorf("Stringify() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("Stringify() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestAnyToString(t *testing.T) {
	tests := []struct {
		name  string
		input interface{}
		want  string
	}{
		{"float64", 3.14, "3.140000"},
		{"int", 42, "42"},
		{"nil", nil, ""},
		{"string", "hello", ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := AnyToString(tt.input)
			if got != tt.want {
				t.Errorf("AnyToString() = %q, want %q", got, tt.want)
			}
		})
	}
}

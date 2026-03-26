package mathutil

import (
	"math"
	"testing"
)

func TestDivide(t *testing.T) {
	tests := []struct {
		name  string
		left  float64
		right float64
		want  float64
	}{
		{"normal division", 10, 3, 10.0 / 3.0},
		{"divide by zero", 10, 0, 0},
		{"zero numerator", 0, 5, 0},
		{"negative numbers", -10, 2, -5},
		{"both zero", 0, 0, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Divide(tt.left, tt.right)
			if math.Abs(got-tt.want) > 1e-10 {
				t.Errorf("Divide(%v, %v) = %v, want %v", tt.left, tt.right, got, tt.want)
			}
		})
	}
}

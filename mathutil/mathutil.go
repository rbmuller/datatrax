// Package mathutil provides safe math operations that handle edge cases
// such as division by zero.
package mathutil

// Divide performs safe float64 division, returning 0 when the divisor is zero
// instead of producing Inf or NaN.
func Divide(left, right float64) float64 {
	if right == 0 {
		return 0
	}
	return left / right
}

// Package dedup provides generic duplicate removal for slices of comparable types.
package dedup

// Deduplicate removes duplicate elements from a slice, preserving order.
// It uses a map to track seen elements for O(n) average time complexity.
func Deduplicate[T comparable](items []T) []T {
	seen := make(map[T]struct{}, len(items))
	result := make([]T, 0, len(items))
	for _, item := range items {
		if _, ok := seen[item]; !ok {
			seen[item] = struct{}{}
			result = append(result, item)
		}
	}
	return result
}

package ml

import "sort"

// LabelEncode converts string labels to numeric values.
// Returns the encoded slice and a mapping from label to numeric value.
func LabelEncode(labels []string) ([]float64, map[string]float64) {
	// Collect unique labels in sorted order for deterministic encoding
	seen := make(map[string]bool)
	var unique []string
	for _, label := range labels {
		if !seen[label] {
			seen[label] = true
			unique = append(unique, label)
		}
	}
	sort.Strings(unique)

	mapping := make(map[string]float64, len(unique))
	for i, label := range unique {
		mapping[label] = float64(i)
	}

	encoded := make([]float64, len(labels))
	for i, label := range labels {
		encoded[i] = mapping[label]
	}
	return encoded, mapping
}

// LabelDecode converts numeric-encoded values back to string labels using
// the mapping produced by LabelEncode.
func LabelDecode(encoded []float64, mapping map[string]float64) []string {
	// Build reverse mapping
	reverse := make(map[float64]string, len(mapping))
	for label, val := range mapping {
		reverse[val] = label
	}

	decoded := make([]string, len(encoded))
	for i, val := range encoded {
		decoded[i] = reverse[val]
	}
	return decoded
}

// OneHotEncode converts string labels into a one-hot encoded matrix.
// Returns the matrix and the sorted list of category names (column order).
func OneHotEncode(labels []string) ([][]float64, []string) {
	// Collect unique labels in sorted order
	seen := make(map[string]bool)
	var categories []string
	for _, label := range labels {
		if !seen[label] {
			seen[label] = true
			categories = append(categories, label)
		}
	}
	sort.Strings(categories)

	catIndex := make(map[string]int, len(categories))
	for i, cat := range categories {
		catIndex[cat] = i
	}

	matrix := make([][]float64, len(labels))
	for i, label := range labels {
		row := make([]float64, len(categories))
		row[catIndex[label]] = 1.0
		matrix[i] = row
	}
	return matrix, categories
}

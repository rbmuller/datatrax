// Package maputil provides utility functions for working with Go maps,
// including JSON unmarshaling and shallow copying.
package maputil

import (
	"encoding/json"
	"fmt"
)

// GenerateMap unmarshals JSON bytes into a map[string]interface{}.
// Returns an error if the JSON is invalid.
func GenerateMap(data []byte) (map[string]interface{}, error) {
	var dataMap map[string]interface{}
	err := json.Unmarshal(data, &dataMap)
	if err != nil {
		return nil, fmt.Errorf("maputil: failed to unmarshal JSON: %w", err)
	}
	return dataMap, nil
}

// CopyMap creates a shallow copy of a map with comparable keys and values.
func CopyMap[K comparable, V any](m map[K]V) map[K]V {
	result := make(map[K]V, len(m))
	for k, v := range m {
		result[k] = v
	}
	return result
}

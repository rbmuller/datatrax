// Package strutil provides string manipulation utilities including trimming,
// joining, splitting, and generic slice membership checks.
package strutil

import (
	"bytes"
	"fmt"
	"regexp"
	"strings"
)

// TrimQuotes removes surrounding double quotes from a string after trimming whitespace.
// If the string does not start and end with double quotes, it is returned as-is.
func TrimQuotes(s string) string {
	s = strings.TrimSpace(s)
	if len(s) >= 2 && s[0] == '"' && s[len(s)-1] == '"' {
		return s[1 : len(s)-1]
	}
	return s
}

// MapArrayToString joins a string slice into a single comma-separated string
// without a trailing comma.
func MapArrayToString(strs []string) string {
	if len(strs) == 0 {
		return ""
	}
	var result bytes.Buffer
	for _, str := range strs {
		result.WriteString(fmt.Sprintf("%s,", str))
	}
	result.Truncate(result.Len() - 1)
	return result.String()
}

// NumbersToString converts a slice of uint values to a delimited string.
// For example, NumbersToString([]uint{1, 2, 3}, ",") returns "1,2,3".
func NumbersToString(nums []uint, delim string) string {
	return strings.Trim(strings.Join(strings.Split(fmt.Sprint(nums), " "), delim), "[]")
}

// StringifyWithQuotes wraps each element in single quotes and joins them
// with commas. Returns an empty string for an empty slice.
func StringifyWithQuotes(items []string) string {
	if len(items) == 0 {
		return ""
	}
	var result bytes.Buffer
	for _, item := range items {
		result.WriteString(fmt.Sprintf("'%s',", item))
	}
	result.Truncate(result.Len() - 1)
	return result.String()
}

// SplitByRegexp splits text using the given regular expression delimiter
// and returns all parts. Panics if the delimiter is not a valid regexp.
func SplitByRegexp(text string, delimiter string) []string {
	reg := regexp.MustCompile(delimiter)
	indexes := reg.FindAllStringIndex(text, -1)
	lastStart := 0
	result := make([]string, len(indexes)+1)

	for i, element := range indexes {
		result[i] = text[lastStart:element[0]]
		lastStart = element[1]
	}
	result[len(indexes)] = text[lastStart:]

	return result
}

// Contains reports whether items contains the given value.
// It replaces type-specific functions like StringInArray, IntInArray, etc.
func Contains[T comparable](items []T, value T) bool {
	for _, item := range items {
		if item == value {
			return true
		}
	}
	return false
}

// SafeIndex returns the element at the given index in list, or the zero value
// of T if the index is out of bounds.
func SafeIndex[T any](list []T, index int) T {
	var zero T
	if index < 0 || index >= len(list) {
		return zero
	}
	return list[index]
}

// AppendIfMissing appends value to the slice only if it is not already present.
func AppendIfMissing[T comparable](slice []T, value T) []T {
	for _, item := range slice {
		if item == value {
			return slice
		}
	}
	return append(slice, value)
}

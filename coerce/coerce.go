// Package coerce provides functions for converting untyped interface{} values
// into concrete Go types with explicit error handling.
package coerce

import (
	"fmt"
	"reflect"
	"strconv"
)

// Floatify converts an interface{} value to float64.
// It supports float64 and string inputs. Returns an error if the conversion fails.
func Floatify(v interface{}) (float64, error) {
	switch i := v.(type) {
	case float64:
		return i, nil
	case float32:
		return float64(i), nil
	case int:
		return float64(i), nil
	case int64:
		return float64(i), nil
	case string:
		f, err := strconv.ParseFloat(i, 64)
		if err != nil {
			return 0, fmt.Errorf("coerce: cannot convert string %q to float64: %w", i, err)
		}
		return f, nil
	case nil:
		return 0, fmt.Errorf("coerce: cannot convert nil to float64")
	default:
		return 0, fmt.Errorf("coerce: unsupported type %T for float64 conversion", v)
	}
}

// Integerify converts an interface{} value to int64.
// It supports float64, int, int64, and string inputs. Returns an error if the conversion fails.
func Integerify(v interface{}) (int64, error) {
	switch i := v.(type) {
	case float64:
		return int64(i), nil
	case float32:
		return int64(i), nil
	case int:
		return int64(i), nil
	case int64:
		return i, nil
	case string:
		n, err := strconv.ParseInt(i, 10, 64)
		if err != nil {
			return 0, fmt.Errorf("coerce: cannot convert string %q to int64: %w", i, err)
		}
		return n, nil
	case nil:
		return 0, fmt.Errorf("coerce: cannot convert nil to int64")
	default:
		return 0, fmt.Errorf("coerce: unsupported type %T for int64 conversion", v)
	}
}

// Boolify converts an interface{} value to bool.
// It accepts bool values directly. Returns an error for non-bool inputs.
func Boolify(v interface{}) (bool, error) {
	switch i := v.(type) {
	case bool:
		return i, nil
	case nil:
		return false, fmt.Errorf("coerce: cannot convert nil to bool")
	default:
		return false, fmt.Errorf("coerce: unsupported type %T for bool conversion", v)
	}
}

// Stringify converts an interface{} value to string.
// It accepts string values directly. Returns an error for nil or non-string inputs.
func Stringify(v interface{}) (string, error) {
	if v == nil {
		return "", fmt.Errorf("coerce: cannot convert nil to string")
	}
	switch i := v.(type) {
	case string:
		return i, nil
	default:
		return "", fmt.Errorf("coerce: unsupported type %T for string conversion", v)
	}
}

// AnyToString converts a numeric interface{} value to its string representation.
// It supports float64 and int types. Returns an empty string for unsupported types.
func AnyToString(v interface{}) string {
	if v == nil {
		return ""
	}
	kind := reflect.TypeOf(v).Kind()
	switch kind {
	case reflect.Float64:
		return fmt.Sprintf("%f", v)
	case reflect.Int:
		return strconv.Itoa(v.(int))
	default:
		return ""
	}
}

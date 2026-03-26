// Package dateutil provides date and time conversion utilities including
// epoch-to-timestamp conversion and date arithmetic.
package dateutil

import (
	"fmt"
	"math"
	"strings"
	"time"
)

// DatePair represents a date range with start and end times.
type DatePair struct {
	Start time.Time
	End   time.Time
}

// EpochToTimestamp converts an epoch time in milliseconds to a formatted timestamp string.
// Returns the formatted timestamp and true on success, or an empty string and false
// if the input is zero.
func EpochToTimestamp(epochMillis int64) (string, bool) {
	if epochMillis == 0 {
		return "", false
	}

	epochSecs := epochMillis / 1000
	t := time.Unix(epochSecs, 0)
	timestamp := t.Format("2006-01-02 15:04:05")

	return timestamp, true
}

// MillisecondsToTime converts a duration in milliseconds to a time.Time value
// measured from the Unix epoch.
func MillisecondsToTime(millis int64) time.Time {
	return time.Unix(0, millis*int64(time.Millisecond))
}

// DaysDifference returns the number of whole days between two times,
// rounded to the nearest integer.
func DaysDifference(start, end time.Time) int {
	return int(math.Round(end.Sub(start).Hours() / 24))
}

// StringToDate parses a date string using the given layout and returns a time.Time.
// Returns the zero value of time.Time and nil error if dateStr is empty.
func StringToDate(layout string, dateStr string) (time.Time, error) {
	if dateStr == "" {
		return time.Time{}, nil
	}
	return time.Parse(layout, dateStr)
}

// CompleteDateWithDays appends "01" to a year-month string (e.g. "2024-01-")
// to form a complete date string.
func CompleteDateWithDays(dateWithoutDay string) string {
	return fmt.Sprintf("%s01", dateWithoutDay)
}

// SplitDateTokens splits a date string in "MM/DD/YYYY" format and returns
// day, month, year components with leading zeros padded to 2 digits.
// Returns "0", "0", "0" if dateStr is empty.
func SplitDateTokens(dateStr string) (day, month, year string) {
	if dateStr == "" {
		return "0", "0", "0"
	}
	tokens := strings.Split(dateStr, "/")
	if len(tokens) < 3 {
		return "0", "0", "0"
	}
	result := make([]string, 0, len(tokens))
	for _, token := range tokens {
		if len(token) == 1 {
			result = append(result, fmt.Sprintf("0%s", token))
		} else {
			result = append(result, token)
		}
	}
	// Input is MM/DD/YYYY, return day, month, year
	return result[1], result[0], result[2]
}

// PadDateWithLeadingZeros takes a date string in "MM/DD/YYYY" format and
// returns it with each component zero-padded. Returns an empty string if
// the input is empty.
func PadDateWithLeadingZeros(dateStr string) string {
	if dateStr == "" {
		return ""
	}
	day, month, year := SplitDateTokens(dateStr)
	return fmt.Sprintf("%s/%s/%s", day, month, year)
}

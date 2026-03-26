package dateutil

import (
	"testing"
	"time"
)

func TestEpochToTimestamp(t *testing.T) {
	tests := []struct {
		name     string
		millis   int64
		wantOK   bool
		wantSub  string
	}{
		{"valid epoch", 1684624830053, true, "2023-05-2"},
		{"zero epoch", 0, false, ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := EpochToTimestamp(tt.millis)
			if ok != tt.wantOK {
				t.Errorf("EpochToTimestamp() ok = %v, want %v", ok, tt.wantOK)
			}
			if tt.wantOK && len(got) == 0 {
				t.Error("EpochToTimestamp() returned empty string for valid input")
			}
		})
	}
}

func TestMillisecondsToTime(t *testing.T) {
	got := MillisecondsToTime(1000)
	if got.Unix() != 1 {
		t.Errorf("MillisecondsToTime(1000) Unix = %d, want 1", got.Unix())
	}
}

func TestDaysDifference(t *testing.T) {
	start := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)
	end := time.Date(2024, 1, 11, 0, 0, 0, 0, time.UTC)
	got := DaysDifference(start, end)
	if got != 10 {
		t.Errorf("DaysDifference() = %d, want 10", got)
	}
}

func TestStringToDate(t *testing.T) {
	got, err := StringToDate("2006-01-02", "2024-03-15")
	if err != nil {
		t.Fatalf("StringToDate() error = %v", err)
	}
	if got.Year() != 2024 || got.Month() != 3 || got.Day() != 15 {
		t.Errorf("StringToDate() = %v, want 2024-03-15", got)
	}
}

func TestStringToDateEmpty(t *testing.T) {
	got, err := StringToDate("2006-01-02", "")
	if err != nil {
		t.Fatalf("StringToDate() error = %v", err)
	}
	if !got.IsZero() {
		t.Errorf("StringToDate(\"\") should return zero time, got %v", got)
	}
}

func TestCompleteDateWithDays(t *testing.T) {
	got := CompleteDateWithDays("2024-01-")
	if got != "2024-01-01" {
		t.Errorf("CompleteDateWithDays() = %q, want \"2024-01-01\"", got)
	}
}

func TestSplitDateTokens(t *testing.T) {
	day, month, year := SplitDateTokens("1/5/2024")
	if day != "05" || month != "01" || year != "2024" {
		t.Errorf("SplitDateTokens() = %s, %s, %s", day, month, year)
	}
}

func TestSplitDateTokensEmpty(t *testing.T) {
	day, month, year := SplitDateTokens("")
	if day != "0" || month != "0" || year != "0" {
		t.Errorf("SplitDateTokens(\"\") = %s, %s, %s", day, month, year)
	}
}

func TestSplitDateTokensTooFewParts(t *testing.T) {
	day, month, year := SplitDateTokens("01/05")
	if day != "0" || month != "0" || year != "0" {
		t.Errorf("SplitDateTokens(\"01/05\") = %s, %s, %s, want 0, 0, 0", day, month, year)
	}
}

func TestSplitDateTokensNoPadding(t *testing.T) {
	day, month, year := SplitDateTokens("12/25/2024")
	if day != "25" || month != "12" || year != "2024" {
		t.Errorf("SplitDateTokens(\"12/25/2024\") = %s, %s, %s", day, month, year)
	}
}

func TestPadDateWithLeadingZeros(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string
	}{
		{"empty string", "", ""},
		{"needs padding", "1/5/2024", "05/01/2024"},
		{"no padding needed", "12/25/2024", "25/12/2024"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := PadDateWithLeadingZeros(tt.input)
			if got != tt.want {
				t.Errorf("PadDateWithLeadingZeros(%q) = %q, want %q", tt.input, got, tt.want)
			}
		})
	}
}

func TestStringToDateInvalidFormat(t *testing.T) {
	_, err := StringToDate("2006-01-02", "not-a-date")
	if err == nil {
		t.Error("StringToDate() expected error for invalid date string")
	}
}

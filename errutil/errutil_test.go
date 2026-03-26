package errutil

import (
	"errors"
	"strings"
	"testing"
)

func TestNewError(t *testing.T) {
	err := NewError(errors.New("test error"))
	if err == nil {
		t.Fatal("NewError() returned nil")
	}
	msg := err.Error()
	if !strings.Contains(msg, "test error") {
		t.Errorf("NewError().Error() = %q, should contain 'test error'", msg)
	}
	if !strings.Contains(msg, ".go:") {
		t.Errorf("NewError().Error() = %q, should contain file info", msg)
	}
}

func TestNewErrorUnwrap(t *testing.T) {
	original := errors.New("original")
	wrapped := NewError(original)
	if !errors.Is(wrapped, original) {
		t.Error("errors.Is() should find original error in chain")
	}
}

func TestNewErrorContainsLineNumber(t *testing.T) {
	err := NewError(errors.New("line check"))
	msg := err.Error()
	// Format is "file.go:LINE - message"
	if !strings.Contains(msg, ".go:") || !strings.Contains(msg, " - ") {
		t.Errorf("error message should have file:line - message format, got %q", msg)
	}
}

func TestNewErrorDifferentMessages(t *testing.T) {
	err1 := NewError(errors.New("first"))
	err2 := NewError(errors.New("second"))
	if err1.Error() == err2.Error() {
		t.Error("different errors should produce different messages")
	}
}

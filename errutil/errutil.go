// Package errutil provides error utilities that capture file and line information
// at the call site for improved debugging.
package errutil

import (
	"fmt"
	"runtime"
)

// callerError is an error that records where it was created.
type callerError struct {
	File string
	Line int
	Err  error
}

// Error returns a string with the file, line, and underlying error message.
func (e *callerError) Error() string {
	return fmt.Sprintf("%s:%d - %s", e.File, e.Line, e.Err.Error())
}

// Unwrap returns the underlying error for use with errors.Is and errors.As.
func (e *callerError) Unwrap() error {
	return e.Err
}

// NewError wraps an error with the caller's file and line information.
// The returned error implements the Unwrap interface for error chain inspection.
func NewError(err error) error {
	_, file, line, _ := runtime.Caller(1)
	return &callerError{
		Err:  err,
		File: file,
		Line: line,
	}
}

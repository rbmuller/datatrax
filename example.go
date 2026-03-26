package main

import (
	"errors"
	"fmt"

	"github.com/rbmuller/datatrax/batch"
	"github.com/rbmuller/datatrax/coerce"
	"github.com/rbmuller/datatrax/dateutil"
	"github.com/rbmuller/datatrax/dedup"
	"github.com/rbmuller/datatrax/errutil"
	"github.com/rbmuller/datatrax/maputil"
	"github.com/rbmuller/datatrax/mathutil"
	"github.com/rbmuller/datatrax/strutil"
)

func main() {
	// Error utilities: wrap errors with caller file and line info
	err := errutil.NewError(errors.New("something went wrong"))
	fmt.Println("Error with location:", err)

	// Dedup: remove duplicates from a slice
	names := []string{"Alice", "Bob", "Alice", "Charlie", "Bob"}
	fmt.Println("Deduplicated:", dedup.Deduplicate(names))

	// Date utilities: convert epoch milliseconds to timestamp
	ts, ok := dateutil.EpochToTimestamp(1684624830053)
	if ok {
		fmt.Println("Timestamp:", ts)
	}

	// Batch: split a slice into chunks
	numbers := []int{1, 2, 3, 4, 5, 6, 7}
	chunks := batch.ChunkArray(numbers, 3)
	fmt.Println("Chunks:", chunks)

	// Coerce: convert interface{} to typed values
	f, _ := coerce.Floatify(3.14)
	fmt.Println("Float:", f)

	// String utilities
	fmt.Println("Contains:", strutil.Contains([]string{"a", "b", "c"}, "b"))
	fmt.Println("TrimQuotes:", strutil.TrimQuotes(`"hello"`))

	// Map utilities: copy a map
	original := map[string]int{"a": 1, "b": 2}
	copied := maputil.CopyMap(original)
	fmt.Println("Copied map:", copied)

	// Math utilities: safe division
	fmt.Println("Safe divide:", mathutil.Divide(10, 3))
	fmt.Println("Divide by zero:", mathutil.Divide(10, 0))
}

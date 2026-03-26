package coerce

import "testing"

func BenchmarkFloatify(b *testing.B) {
	inputs := []interface{}{
		42.5,
		float32(3.14),
		100,
		int64(9999),
		"123.456",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, v := range inputs {
			Floatify(v)
		}
	}
}

func BenchmarkStringify(b *testing.B) {
	inputs := []interface{}{
		"hello",
		"world",
		"benchmark-test-string",
		"another value",
		"last one",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, v := range inputs {
			Stringify(v)
		}
	}
}

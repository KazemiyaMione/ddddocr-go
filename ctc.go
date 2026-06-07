package ddddocr

import (
	"math"
)

// ctcDecode performs CTC decoding on logits.
//
// Input: logits (flat float32 array), shape = (seqLen, 1, numClasses)
// That is: for each timestep t in [0, seqLen), the class probabilities are at
// logits[t*numClasses : (t+1)*numClasses].
//
// The batch dimension is always 1, so it's effectively (seqLen, numClasses).
//
// Steps:
//  1. argmax along class axis for each timestep
//  2. Remove consecutive duplicates
//  3. Skip blank label (index 0)
//  4. Map indices to charset characters
func ctcDecode(logits []float32, seqLen, numClasses int, charset []string) string {
	if seqLen == 0 || numClasses == 0 {
		return ""
	}

	// Step 1: argmax per timestep
	indices := make([]int, seqLen)
	for t := 0; t < seqLen; t++ {
		offset := t * numClasses
		maxVal := float32(math.Inf(-1))
		maxIdx := 0
		for c := 0; c < numClasses; c++ {
			val := logits[offset+c]
			if val > maxVal {
				maxVal = val
				maxIdx = c
			}
		}
		indices[t] = maxIdx
	}

	// Step 2+3: CTC decode — remove consecutive duplicates and skip blank (index 0)
	var decoded []int
	var prev int = -1 // sentinel different from any valid index
	for _, idx := range indices {
		if idx == prev {
			continue // skip consecutive duplicate
		}
		prev = idx
		if idx == 0 {
			continue // skip blank
		}
		decoded = append(decoded, idx)
	}

	// Step 4: Map indices to characters
	return indicesToText(decoded, charset)
}

// ctcDecodeWithConfidence performs CTC decoding and also returns per-character confidence scores.
func ctcDecodeWithConfidence(logits []float32, seqLen, numClasses int, charset []string) (string, []float32) {
	if seqLen == 0 || numClasses == 0 {
		return "", nil
	}

	// Step 1: argmax + max probability per timestep
	indices := make([]int, seqLen)
	confs := make([]float32, seqLen)
	for t := 0; t < seqLen; t++ {
		offset := t * numClasses
		maxVal := float32(math.Inf(-1))
		maxIdx := 0
		for c := 0; c < numClasses; c++ {
			val := logits[offset+c]
			if val > maxVal {
				maxVal = val
				maxIdx = c
			}
		}
		indices[t] = maxIdx
		confs[t] = maxVal
	}

	// Step 2+3: CTC decode
	var decoded []int
	var decodedConfs []float32
	var prev int = -1
	for i, idx := range indices {
		if idx == prev {
			continue
		}
		prev = idx
		if idx == 0 {
			continue
		}
		decoded = append(decoded, idx)
		decodedConfs = append(decodedConfs, confs[i])
	}

	return indicesToText(decoded, charset), decodedConfs
}

// indicesToText converts a list of charset indices to a string.
func indicesToText(indices []int, charset []string) string {
	var result []rune
	for _, idx := range indices {
		if idx >= 0 && idx < len(charset) {
			char := charset[idx]
			if char != "" {
				result = append(result, []rune(char)...)
			}
		}
	}
	return string(result)
}

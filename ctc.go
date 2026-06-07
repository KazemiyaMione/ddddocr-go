package ddddocr

import (
	"math"
)

// ctcDecode 对 ONNX 输出的 logits 进行 CTC 解码。
//
// 输入：logits 为一维 float32 数组，逻辑形状为 (seqLen, 1, numClasses)，
// 即第 t 个时间步的类别概率存储在 logits[t*numClasses : (t+1)*numClasses]。
// batch 维度固定为 1，因此实际等价于 (seqLen, numClasses)。
//
// 步骤：
//  1. 每个时间步取 argmax（概率最大的类别索引）
//  2. 去除连续重复的索引
//  3. 跳过空白标签（索引 0）
//  4. 将索引映射为字符集字符
func ctcDecode(logits []float32, seqLen, numClasses int, charset []string) string {
	if seqLen == 0 || numClasses == 0 {
		return ""
	}

	// 第一步：每个时间步取 argmax
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

	// 第二、三步：CTC 解码 —— 去重并跳过空白
	var decoded []int
	var prev int = -1
	for _, idx := range indices {
		if idx == prev {
			continue // 跳过连续重复
		}
		prev = idx
		if idx == 0 {
			continue // 跳过空白
		}
		decoded = append(decoded, idx)
	}

	// 第四步：索引映射为字符
	return indicesToText(decoded, charset)
}

// ctcDecodeWithConfidence 执行 CTC 解码并返回每个字符的置信度。
func ctcDecodeWithConfidence(logits []float32, seqLen, numClasses int, charset []string) (string, []float32) {
	if seqLen == 0 || numClasses == 0 {
		return "", nil
	}

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

// indicesToText 将字符集索引列表转换为字符串。
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

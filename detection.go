package ddddocr

import (
	"bytes"
	_ "embed"
	"fmt"
	"image"
	"math"
	"sort"

	"golang.org/x/image/draw"

	ort "github.com/yalue/onnxruntime_go"
)

//go:embed models/common_det.onnx
var commonDetOnnxData []byte

// nmsCandidate 表示 NMS 候选框。
type nmsCandidate struct {
	box   [4]float64
	score float64
}

// Detection 对图片进行目标检测，返回检测到的边界框列表。
// 每个边界框格式为 [x1, y1, x2, y2]（左上角和右下角坐标）。
func (ocr *DdddOcr) Detection(imgData []byte) ([][4]int, error) {
	// 解码图片
	img, _, err := image.Decode(bytes.NewReader(imgData))
	if err != nil {
		return nil, fmt.Errorf("图片解码失败: %w", err)
	}

	// 预处理：缩放 + 填充到 416x416
	inputData, ratio := preprocDet(img)

	// 运行 ONNX 推理
	outputData, err := runDetInference(ocr.detSession, inputData)
	if err != nil {
		return nil, fmt.Errorf("检测推理失败: %w", err)
	}

	// 后处理：解码 + NMS
	boxes := postprocDet(outputData, ratio, img.Bounds().Dx(), img.Bounds().Dy())

	return boxes, nil
}

// preprocDet 图片预处理：等比缩放到 416x416，不足部分用 114 填充。
func preprocDet(img image.Image) ([]float32, float64) {
	const inputSize = 416

	origW := img.Bounds().Dx()
	origH := img.Bounds().Dy()

	// 计算缩放比例
	r := float64(inputSize) / float64(max(origH, origW))

	newW := int(float64(origW) * r)
	newH := int(float64(origH) * r)

	// 缩放图片（Bilinear，对应 Python 的 cv2.INTER_LINEAR）
	resized := image.NewRGBA(image.Rect(0, 0, newW, newH))
	draw.BiLinear.Scale(resized, resized.Bounds(), img, img.Bounds(), draw.Over, nil)

	// 创建 416x416 填充画布（CHW 顺序，填充值 114）
	padded := make([]float32, 3*inputSize*inputSize)
	for c := 0; c < 3; c++ {
		offset := c * inputSize * inputSize
		for i := 0; i < inputSize*inputSize; i++ {
			padded[offset+i] = 114.0
		}
	}

	// 将缩放后的图片复制到填充画布（BGR 顺序，对应 Python 的 cv2.imdecode）
	for y := 0; y < newH; y++ {
		for x := 0; x < newW; x++ {
			cr, cg, cb, _ := resized.At(x, y).RGBA()
			padded[0*inputSize*inputSize+y*inputSize+x] = float32(cb >> 8) // B
			padded[1*inputSize*inputSize+y*inputSize+x] = float32(cg >> 8) // G
			padded[2*inputSize*inputSize+y*inputSize+x] = float32(cr >> 8) // R
		}
	}

	return padded, r
}

// runDetInference 运行检测模型推理。
func runDetInference(session *ort.DynamicAdvancedSession, inputData []float32) ([]float32, error) {
	inputShape := ort.NewShape(1, 3, 416, 416)
	inputTensor, err := ort.NewTensor(inputShape, inputData)
	if err != nil {
		return nil, fmt.Errorf("创建输入张量失败: %w", err)
	}
	defer inputTensor.Destroy()

	// 输出形状: (1, 3549, 6)
	outputShape := ort.NewShape(1, 3549, 6)
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		return nil, fmt.Errorf("创建输出张量失败: %w", err)
	}
	defer outputTensor.Destroy()

	err = session.Run(
		[]ort.Value{inputTensor},
		[]ort.Value{outputTensor},
	)
	if err != nil {
		return nil, fmt.Errorf("ONNX 推理失败: %w", err)
	}

	return outputTensor.GetData(), nil
}

// postprocDet 检测后处理：YOLOX 风格解码 + NMS。
func postprocDet(output []float32, ratio float64, origW, origH int) [][4]int {
	const (
		inputSize  = 416
		nmsThr     = 0.45
		scoreThr   = 0.1
		stridesLen = 3
	)

	strides := [3]int{8, 16, 32}
	hsizes := [3]int{inputSize / 8, inputSize / 16, inputSize / 32}
	wsizes := [3]int{inputSize / 8, inputSize / 16, inputSize / 32}

	// 复制输出数据以便原地修改
	pred := make([]float32, len(output))
	copy(pred, output)

	// 解码预测框
	offset := 0
	for i := 0; i < stridesLen; i++ {
		hsize := hsizes[i]
		wsize := wsizes[i]
		stride := strides[i]
		n := hsize * wsize

		for j := 0; j < n; j++ {
			idx := offset + j*6

			gx := j % wsize
			gy := j / wsize

			pred[idx+0] = (pred[idx+0] + float32(gx)) * float32(stride)
			pred[idx+1] = (pred[idx+1] + float32(gy)) * float32(stride)
			pred[idx+2] = float32(math.Exp(float64(pred[idx+2]))) * float32(stride)
			pred[idx+3] = float32(math.Exp(float64(pred[idx+3]))) * float32(stride)
		}
		offset += n * 6
	}

	// 收集候选框
	var candidates []nmsCandidate
	offset = 0
	for i := 0; i < stridesLen; i++ {
		n := hsizes[i] * wsizes[i]
		for j := 0; j < n; j++ {
			idx := offset + j*6

			cx := float64(pred[idx+0])
			cy := float64(pred[idx+1])
			w := float64(pred[idx+2])
			h := float64(pred[idx+3])
			objScore := float64(pred[idx+4])
			clsScore := float64(pred[idx+5])

			score := objScore * clsScore
			if score < scoreThr {
				continue
			}

			x1 := (cx - w/2) / ratio
			y1 := (cy - h/2) / ratio
			x2 := (cx + w/2) / ratio
			y2 := (cy + h/2) / ratio

			candidates = append(candidates, nmsCandidate{
				box:   [4]float64{x1, y1, x2, y2},
				score: score,
			})
		}
		offset += n * 6
	}

	// NMS 非极大值抑制
	keep := nmsFilter(candidates, nmsThr)

	// 转换为整数边界框
	var result [][4]int
	for _, idx := range keep {
		b := candidates[idx].box
		x1 := clampInt(b[0], 0, origW)
		y1 := clampInt(b[1], 0, origH)
		x2 := clampInt(b[2], 0, origW)
		y2 := clampInt(b[3], 0, origH)
		result = append(result, [4]int{x1, y1, x2, y2})
	}

	return result
}

// nmsFilter 类别无关的非极大值抑制。
func nmsFilter(candidates []nmsCandidate, threshold float64) []int {
	if len(candidates) == 0 {
		return nil
	}

	// 按分数降序排列
	indices := make([]int, len(candidates))
	for i := range indices {
		indices[i] = i
	}
	sort.Slice(indices, func(i, j int) bool {
		return candidates[indices[i]].score > candidates[indices[j]].score
	})

	var keep []int
	for len(indices) > 0 {
		best := indices[0]
		keep = append(keep, best)
		indices = indices[1:]

		bestBox := candidates[best].box
		bestArea := (bestBox[2] - bestBox[0] + 1) * (bestBox[3] - bestBox[1] + 1)

		var remaining []int
		for _, idx := range indices {
			box := candidates[idx].box

			xx1 := maxf(bestBox[0], box[0])
			yy1 := maxf(bestBox[1], box[1])
			xx2 := minf(bestBox[2], box[2])
			yy2 := minf(bestBox[3], box[3])

			iw := maxf(0, xx2-xx1+1)
			ih := maxf(0, yy2-yy1+1)
			inter := iw * ih

			area := (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
			iou := inter / (bestArea + area - inter)

			if iou <= threshold {
				remaining = append(remaining, idx)
			}
		}
		indices = remaining
	}

	return keep
}

func clampInt(v float64, lo, hi int) int {
	if v < float64(lo) {
		return lo
	}
	if v > float64(hi) {
		return hi
	}
	return int(v)
}

func maxf(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}

func minf(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}

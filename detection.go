package ddddocr

import (
	"bytes"
	_ "embed"
	"fmt"
	"image"
	"image/color"
	"math"
	"sort"

	ort "github.com/yalue/onnxruntime_go"
)

//go:embed models/common_det.onnx
var commonDetOnnxData []byte

// nmsCandidate 表示 NMS 候选框。
type nmsCandidate struct {
	box   [4]float64
	score float64
}

// DetectionDebug 调试用：返回原始模型输出和预处理数据以供对比。
func (ocr *DdddOcr) DetectionDebug(imgData []byte) (input []float32, output []float32, ratio float64, origW, origH int, err error) {
	img, _, e := image.Decode(bytes.NewReader(imgData))
	if e != nil {
		err = fmt.Errorf("图片解码失败: %w", e)
		return
	}

	input, ratio = preprocDet(img)
	origW = img.Bounds().Dx()
	origH = img.Bounds().Dy()
	output, err = runDetInference(ocr.detSession, input)
	return
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

	r := float64(inputSize) / float64(max(origH, origW))

	newW := int(float64(origW) * r)
	newH := int(float64(origH) * r)

	// 手动 Bilinear，对齐 OpenCV cv2.INTER_LINEAR 的半像素偏移
	resized := resizeBilinearOCV(img, newW, newH)

	// 创建 416x416 填充画布（CHW 顺序，填充值 114）
	padded := make([]float32, 3*inputSize*inputSize)
	for i := range padded {
		padded[i] = 114.0
	}

	// BGR 顺序贴入（匹配 Python cv2.imdecode 输出）
	for y := 0; y < newH; y++ {
		rowOff := y * inputSize
		for x := 0; x < newW; x++ {
			cr, cg, cb, _ := resized.At(x, y).RGBA()
			padded[0*inputSize*inputSize+rowOff+x] = float32(cb >> 8) // B
			padded[1*inputSize*inputSize+rowOff+x] = float32(cg >> 8) // G
			padded[2*inputSize*inputSize+rowOff+x] = float32(cr >> 8) // R
		}
	}

	return padded, r
}

// resizeBilinearOCV 手动 bilinear 缩放，使用 OpenCV 的半像素偏移公式：
//
//	src_x = (dst_x + 0.5) / scale - 0.5
//	src_y = (dst_y + 0.5) / scale - 0.5
//
// 超出边界时 clamp 到边沿像素。
func resizeBilinearOCV(img image.Image, dstW, dstH int) *image.RGBA {
	bounds := img.Bounds()
	srcW := bounds.Dx()
	srcH := bounds.Dy()
	offX := bounds.Min.X
	offY := bounds.Min.Y

	scaleX := float64(srcW) / float64(dstW)
	scaleY := float64(srcH) / float64(dstH)

	dst := image.NewRGBA(image.Rect(0, 0, dstW, dstH))

	for dy := 0; dy < dstH; dy++ {
		sy := (float64(dy)+0.5)*scaleY - 0.5
		sy0 := int(math.Floor(sy))
		sy1 := sy0 + 1
		if sy0 < 0 {
			sy0 = 0
		}
		if sy1 >= srcH {
			sy1 = srcH - 1
		}
		fy := sy - float64(sy0)

		for dx := 0; dx < dstW; dx++ {
			sx := (float64(dx)+0.5)*scaleX - 0.5
			sx0 := int(math.Floor(sx))
			sx1 := sx0 + 1
			if sx0 < 0 {
				sx0 = 0
			}
			if sx1 >= srcW {
				sx1 = srcW - 1
			}
			fx := sx - float64(sx0)

			r00, g00, b00, _ := img.At(sx0+offX, sy0+offY).RGBA()
			r10, g10, b10, _ := img.At(sx1+offX, sy0+offY).RGBA()
			r01, g01, b01, _ := img.At(sx0+offX, sy1+offY).RGBA()
			r11, g11, b11, _ := img.At(sx1+offX, sy1+offY).RGBA()

			topR := (1-fx)*float64(r00) + fx*float64(r10)
			botR := (1-fx)*float64(r01) + fx*float64(r11)
			topG := (1-fx)*float64(g00) + fx*float64(g10)
			botG := (1-fx)*float64(g01) + fx*float64(g11)
			topB := (1-fx)*float64(b00) + fx*float64(b10)
			botB := (1-fx)*float64(b01) + fx*float64(b11)

			rv := uint8(uint16(clampF((1-fy)*topR+fy*botR, 0, 65535)) >> 8)
			gv := uint8(uint16(clampF((1-fy)*topG+fy*botG, 0, 65535)) >> 8)
			bv := uint8(uint16(clampF((1-fy)*topB+fy*botB, 0, 65535)) >> 8)

			dst.SetRGBA(dx, dy, color.RGBA{R: rv, G: gv, B: bv, A: 255})
		}
	}

	return dst
}

func clampF(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
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

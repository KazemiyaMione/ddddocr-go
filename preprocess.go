package ddddocr

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"

	_ "golang.org/x/image/bmp"
	_ "golang.org/x/image/tiff"
	_ "image/gif"

	"golang.org/x/image/draw"
)

// decodeImage 将图片字节数据解码为 image.Image，支持 PNG、JPEG、BMP、GIF、TIFF。
func decodeImage(data []byte) (image.Image, error) {
	img, format, err := image.Decode(bytes.NewReader(data))
	if err == nil {
		_ = format
		return img, nil
	}
	return nil, fmt.Errorf("不支持的图片格式或数据损坏: %w", err)
}

// preprocess 将图片转换为 ONNX 推理所需的 float32 NCHW 张量。
//
// 步骤：
//  1. 转为灰度图（亮度）
//  2. 缩放：高度=64，宽度 = 原始宽度 × 64 / 原始高度（双线性插值）
//  3. 归一化：除以 255.0 → [0, 1]
//
// 返回 (数据, 高度, 宽度, 错误)。
func preprocess(img image.Image) ([]float32, int, int, error) {
	bounds := img.Bounds()
	origW := bounds.Dx()
	origH := bounds.Dy()

	if origW <= 0 || origH <= 0 {
		return nil, 0, 0, fmt.Errorf("无效的图片尺寸: %dx%d", origW, origH)
	}

	targetH := 64
	targetW := origW * targetH / origH
	if targetW < 1 {
		targetW = 1
	}

	// 缩放并转为灰度
	resized := image.NewGray(image.Rect(0, 0, targetW, targetH))
	draw.CatmullRom.Scale(resized, resized.Bounds(), img, img.Bounds(), draw.Over, nil)

	// 构建 NCHW 张量: (1, 1, 64, targetW)
	total := targetH * targetW
	data := make([]float32, total)

	for y := 0; y < targetH; y++ {
		rowOffset := y * targetW
		for x := 0; x < targetW; x++ {
			gray := resized.GrayAt(x, y)
			data[rowOffset+x] = float32(gray.Y) / 255.0
		}
	}

	return data, targetH, targetW, nil
}

// preprocessWithPNGFix 处理图片，可选 RGBA 透明背景修复。
func preprocessWithPNGFix(img image.Image, pngFix bool) ([]float32, int, int, error) {
	if pngFix {
		if hasAlpha(img) {
			img = compositeOverWhite(img)
		}
	}
	return preprocess(img)
}

// hasAlpha 检查图片是否含有透明通道。
func hasAlpha(img image.Image) bool {
	switch img.(type) {
	case *image.NRGBA, *image.RGBA, *image.NYCbCrA:
		return true
	default:
		return false
	}
}

// compositeOverWhite 将图片合成到白色背景上。
func compositeOverWhite(img image.Image) image.Image {
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()

	result := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			result.Set(x, y, color.White)
		}
	}
	draw.Over.Draw(result, result.Bounds(), img, bounds.Min)

	return result
}

// imageToGray 将任意图片转为灰度图。
func imageToGray(img image.Image) *image.Gray {
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	gray := image.NewGray(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			gray.Set(x, y, img.At(x, y))
		}
	}
	return gray
}

func init() {
	image.RegisterFormat("png", "\x89PNG\r\n\x1a\n", png.Decode, png.DecodeConfig)
	image.RegisterFormat("jpeg", "\xff\xd8", jpeg.Decode, jpeg.DecodeConfig)
}

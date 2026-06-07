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

// decodeImage decodes image bytes into an image.Image, supporting PNG, JPEG, BMP, GIF, TIFF.
func decodeImage(data []byte) (image.Image, error) {
	// Try standard library formats first
	img, format, err := image.Decode(bytes.NewReader(data))
	if err == nil {
		_ = format
		return img, nil
	}

	// Fallback: try to re-decode from a fresh reader (some formats need seeking)
	return nil, fmt.Errorf("unsupported image format or corrupted data: %w", err)
}

// preprocess converts an image to a float32 NCHW tensor for ONNX inference.
//
// Steps:
//  1. Convert to grayscale (luminance)
//  2. Resize: height=64, width = img.Width * 64 / img.Height (bilinear)
//  3. Normalize: divide by 255.0 → [0, 1]
//
// Returns (data, height, width, error).
func preprocess(img image.Image) ([]float32, int, int, error) {
	bounds := img.Bounds()
	origW := bounds.Dx()
	origH := bounds.Dy()

	if origW <= 0 || origH <= 0 {
		return nil, 0, 0, fmt.Errorf("invalid image dimensions: %dx%d", origW, origH)
	}

	// Calculate target dimensions
	targetH := 64
	targetW := origW * targetH / origH
	if targetW < 1 {
		targetW = 1
	}

	// Resize using bilinear interpolation (draw.CatmullRom ≈ bilinear quality)
	resized := image.NewGray(image.Rect(0, 0, targetW, targetH))
	draw.CatmullRom.Scale(resized, resized.Bounds(), img, img.Bounds(), draw.Over, nil)

	// Build NCHW tensor: (1, 1, 64, targetW)
	// Total elements = 1 * 1 * targetH * targetW
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

// preprocessWithPNGFix handles RGBA images by compositing over white background.
// This matches the Python png_rgba_black_preprocess behavior.
func preprocessWithPNGFix(img image.Image, pngFix bool) ([]float32, int, int, error) {
	if pngFix {
		// If image has alpha channel, composite onto white background
		if hasAlpha(img) {
			img = compositeOverWhite(img)
		}
	}
	return preprocess(img)
}

// hasAlpha checks if an image has transparency (NRGBA, RGBA, etc.)
func hasAlpha(img image.Image) bool {
	switch img.(type) {
	case *image.NRGBA, *image.RGBA, *image.NYCbCrA, *image.Gray16: // Gray16 not alpha but rare
		return true
	default:
		return false
	}
}

// compositeOverWhite composites the image over a white background.
func compositeOverWhite(img image.Image) image.Image {
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()

	// Create a white RGBA image
	result := image.NewRGBA(image.Rect(0, 0, w, h))

	// Fill with white
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			result.Set(x, y, color.White)
		}
	}

	// Draw original image on top
	draw.Over.Draw(result, result.Bounds(), img, bounds.Min)

	return result
}

// imageToGray converts any image to a grayscale image.
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
	// Register additional image formats
	image.RegisterFormat("png", "\x89PNG\r\n\x1a\n", png.Decode, png.DecodeConfig)
	image.RegisterFormat("jpeg", "\xff\xd8", jpeg.Decode, jpeg.DecodeConfig)
}

package ddddocr

import (
	"fmt"
	"image"
	"image/color"
	"math"
)

// SlideMatchResult holds the result of slider matching.
type SlideMatchResult struct {
	Target     [2]int   `json:"target"`     // [x, y] center coordinates
	TargetX    int      `json:"target_x"`   // x center coordinate
	TargetY    int      `json:"target_y"`   // y center coordinate
	Confidence float64  `json:"confidence"` // match confidence (0-1)
}

// SlideMatch performs slider captcha matching.
//
// targetImage: the slider piece image (bytes, PNG/JPEG/etc.)
// backgroundImage: the background with gap (bytes)
// simpleTarget: if true, uses direct template matching; if false, uses edge-based matching (default).
//
// Returns the match result with the target center coordinates.
func (ocr *DdddOcr) SlideMatch(targetImage, backgroundImage []byte, simpleTarget bool) (*SlideMatchResult, error) {
	target, err := decodeImage(targetImage)
	if err != nil {
		return nil, fmt.Errorf("failed to decode target image: %w", err)
	}
	background, err := decodeImage(backgroundImage)
	if err != nil {
		return nil, fmt.Errorf("failed to decode background image: %w", err)
	}

	targetGray := imageToGray(target)
	backgroundGray := imageToGray(background)

	if simpleTarget {
		return simpleTemplateMatch(targetGray, backgroundGray)
	}
	return edgeBasedMatch(targetGray, backgroundGray)
}

// SlideComparison performs slider gap comparison.
//
// targetImage: image with the gap/pit (bytes)
// backgroundImage: complete background image without gap (bytes)
//
// Finds the gap by computing the difference between the two images.
func (ocr *DdddOcr) SlideComparison(targetImage, backgroundImage []byte) (*SlideMatchResult, error) {
	img1, err := decodeImage(targetImage)
	if err != nil {
		return nil, fmt.Errorf("failed to decode target image: %w", err)
	}
	img2, err := decodeImage(backgroundImage)
	if err != nil {
		return nil, fmt.Errorf("failed to decode background image: %w", err)
	}

	return diffBasedComparison(img1, img2)
}

// simpleTemplateMatch performs direct normalized cross-correlation template matching.
func simpleTemplateMatch(target, background *image.Gray) (*SlideMatchResult, error) {
	bgW, bgH := background.Bounds().Dx(), background.Bounds().Dy()
	tW, tH := target.Bounds().Dx(), target.Bounds().Dy()

	if tW > bgW || tH > bgH {
		return nil, fmt.Errorf("target image (%dx%d) larger than background (%dx%d)", tW, tH, bgW, bgH)
	}

	bgPix := float64Slice(background.Pix, bgW, bgH)
	tPix := float64Slice(target.Pix, tW, tH)

	bestX, bestY, bestScore := nccMatch(bgPix, bgW, bgH, tPix, tW, tH)

	centerX := bestX + tW/2
	centerY := bestY + tH/2

	return &SlideMatchResult{
		Target:     [2]int{centerX, centerY},
		TargetX:    centerX,
		TargetY:    centerY,
		Confidence: bestScore,
	}, nil
}

// edgeBasedMatch performs Canny-like edge detection then template matching.
func edgeBasedMatch(target, background *image.Gray) (*SlideMatchResult, error) {
	targetEdges := sobelEdges(target)
	backgroundEdges := sobelEdges(background)

	return simpleTemplateMatch(targetEdges, backgroundEdges)
}

// diffBasedComparison finds the gap by computing image difference.
func diffBasedComparison(img1, img2 image.Image) (*SlideMatchResult, error) {
	bounds1 := img1.Bounds()
	bounds2 := img2.Bounds()

	w := min(bounds1.Dx(), bounds2.Dx())
	h := min(bounds1.Dy(), bounds2.Dy())

	// Compute absolute difference in grayscale
	diff := image.NewGray(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			g1 := luminance(img1.At(x+bounds1.Min.X, y+bounds1.Min.Y))
			g2 := luminance(img2.At(x+bounds2.Min.X, y+bounds2.Min.Y))
			d := abs(g1 - g2)
			diff.SetGray(x, y, color.Gray{Y: uint8(d)})
		}
	}

	// Binary threshold at 30
	binary := image.NewGray(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			if diff.GrayAt(x, y).Y > 30 {
				binary.SetGray(x, y, color.Gray{Y: 255})
			} else {
				binary.SetGray(x, y, color.Gray{Y: 0})
			}
		}
	}

	// Morphological close + open (3x3 kernel)
	binary = morphologyClose(binary, 3)
	binary = morphologyOpen(binary, 3)

	// Find the largest connected component (contour)
	centerX, centerY := findLargestBlobCenter(binary)

	return &SlideMatchResult{
		Target:  [2]int{centerX, centerY},
		TargetX: centerX,
		TargetY: centerY,
	}, nil
}

// ------------------- image processing utilities -------------------

// float64Slice extracts pixel data as float64 slice.
func float64Slice(pix []uint8, w, h int) []float64 {
	out := make([]float64, w*h)
	for i := 0; i < w*h && i < len(pix); i++ {
		out[i] = float64(pix[i])
	}
	return out
}

// luminance converts a color to grayscale luminance value (0-255).
func luminance(c color.Color) int {
	r, g, b, _ := c.RGBA()
	// ITU-R BT.601
	return int(0.299*float64(r>>8) + 0.587*float64(g>>8) + 0.114*float64(b>>8))
}

// nccMatch performs normalized cross-correlation template matching.
// Returns best match (x, y, score).
func nccMatch(bg []float64, bgW, bgH int, tmpl []float64, tW, tH int) (int, int, float64) {
	// Precompute template mean and stddev
	tMean := mean(tmpl)
	tStd := stddev(tmpl, tMean)
	if tStd < 1e-10 {
		tStd = 1.0
	}

	bestX, bestY := 0, 0
	bestScore := -1.0

	for y := 0; y <= bgH-tH; y++ {
		for x := 0; x <= bgW-tW; x++ {
			score := nccScore(bg, bgW, x, y, tmpl, tW, tH, tMean, tStd)
			if score > bestScore {
				bestScore = score
				bestX = x
				bestY = y
			}
		}
	}

	// Normalize score to [0, 1]
	if bestScore > 1.0 {
		bestScore = 1.0
	}
	if bestScore < 0 {
		bestScore = 0
	}

	return bestX, bestY, bestScore
}

// nccScore computes NCC score at a specific position.
func nccScore(bg []float64, bgW, x, y int, tmpl []float64, tW, tH int, tMean, tStd float64) float64 {
	// Extract window from background
	window := make([]float64, tW*tH)
	for wy := 0; wy < tH; wy++ {
		for wx := 0; wx < tW; wx++ {
			window[wy*tW+wx] = bg[(y+wy)*bgW+(x+wx)]
		}
	}

	wMean := mean(window)
	wStd := stddev(window, wMean)
	if wStd < 1e-10 {
		return 0
	}

	// Compute normalized cross-correlation
	var sum float64
	for i := 0; i < len(tmpl); i++ {
		sum += ((tmpl[i] - tMean) / tStd) * ((window[i] - wMean) / wStd)
	}

	return sum / float64(len(tmpl))
}

func mean(data []float64) float64 {
	var sum float64
	for _, v := range data {
		sum += v
	}
	return sum / float64(len(data))
}

func stddev(data []float64, m float64) float64 {
	var sum float64
	for _, v := range data {
		d := v - m
		sum += d * d
	}
	return math.Sqrt(sum / float64(len(data)))
}

// sobelEdges performs Sobel edge detection on a grayscale image.
func sobelEdges(img *image.Gray) *image.Gray {
	w, h := img.Bounds().Dx(), img.Bounds().Dy()
	edges := image.NewGray(image.Rect(0, 0, w, h))

	// Sobel kernels
	gx := [3][3]int{{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}}
	gy := [3][3]int{{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}}

	for y := 1; y < h-1; y++ {
		for x := 1; x < w-1; x++ {
			var sumX, sumY int
			for ky := -1; ky <= 1; ky++ {
				for kx := -1; kx <= 1; kx++ {
					val := int(img.GrayAt(x+kx, y+ky).Y)
					sumX += val * gx[ky+1][kx+1]
					sumY += val * gy[ky+1][kx+1]
				}
			}
			mag := int(math.Sqrt(float64(sumX*sumX + sumY*sumY)))
			if mag > 255 {
				mag = 255
			}
			edges.SetGray(x, y, color.Gray{Y: uint8(mag)})
		}
	}

	return edges
}

// morphologyClose performs morphological closing (dilate then erode).
func morphologyClose(img *image.Gray, kernelSize int) *image.Gray {
	return morphologyErode(morphologyDilate(img, kernelSize), kernelSize)
}

// morphologyOpen performs morphological opening (erode then dilate).
func morphologyOpen(img *image.Gray, kernelSize int) *image.Gray {
	return morphologyDilate(morphologyErode(img, kernelSize), kernelSize)
}

func morphologyDilate(img *image.Gray, kernelSize int) *image.Gray {
	w, h := img.Bounds().Dx(), img.Bounds().Dy()
	result := image.NewGray(image.Rect(0, 0, w, h))
	offset := kernelSize / 2

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			var maxVal uint8
			for ky := -offset; ky <= offset; ky++ {
				for kx := -offset; kx <= offset; kx++ {
					nx, ny := x+kx, y+ky
					if nx >= 0 && nx < w && ny >= 0 && ny < h {
						if img.GrayAt(nx, ny).Y > maxVal {
							maxVal = img.GrayAt(nx, ny).Y
						}
					}
				}
			}
			result.SetGray(x, y, color.Gray{Y: maxVal})
		}
	}
	return result
}

func morphologyErode(img *image.Gray, kernelSize int) *image.Gray {
	w, h := img.Bounds().Dx(), img.Bounds().Dy()
	result := image.NewGray(image.Rect(0, 0, w, h))
	offset := kernelSize / 2

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			var minVal uint8 = 255
			for ky := -offset; ky <= offset; ky++ {
				for kx := -offset; kx <= offset; kx++ {
					nx, ny := x+kx, y+ky
					if nx >= 0 && nx < w && ny >= 0 && ny < h {
						if img.GrayAt(nx, ny).Y < minVal {
							minVal = img.GrayAt(nx, ny).Y
						}
					}
				}
			}
			result.SetGray(x, y, color.Gray{Y: minVal})
		}
	}
	return result
}

// findLargestBlobCenter finds the center of the largest white blob in a binary image.
func findLargestBlobCenter(img *image.Gray) (int, int) {
	w, h := img.Bounds().Dx(), img.Bounds().Dy()
	visited := make([]bool, w*h)

	bestArea := 0
	bestCX, bestCY := 0, 0

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			if img.GrayAt(x, y).Y > 128 && !visited[y*w+x] {
				// BFS to find connected component
				area, sumX, sumY := bfsBlob(img, visited, w, h, x, y)
				if area > bestArea {
					bestArea = area
					bestCX = sumX / area
					bestCY = sumY / area
				}
			}
		}
	}

	return bestCX, bestCY
}

// bfsBlob performs BFS to find a connected white component.
func bfsBlob(img *image.Gray, visited []bool, w, h, startX, startY int) (area, sumX, sumY int) {
	type point struct{ x, y int }
	queue := []point{{startX, startY}}
	visited[startY*w+startX] = true

	dirs := [][2]int{{0, 1}, {1, 0}, {0, -1}, {-1, 0}}

	for len(queue) > 0 {
		p := queue[0]
		queue = queue[1:]
		area++
		sumX += p.x
		sumY += p.y

		for _, d := range dirs {
			nx, ny := p.x+d[0], p.y+d[1]
			if nx >= 0 && nx < w && ny >= 0 && ny < h {
				if !visited[ny*w+nx] && img.GrayAt(nx, ny).Y > 128 {
					visited[ny*w+nx] = true
					queue = append(queue, point{nx, ny})
				}
			}
		}
	}
	return
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

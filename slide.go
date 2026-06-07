package ddddocr

import (
	"fmt"
	"image"
	"image/color"
	"math"
)

// SlideMatchResult 滑块匹配结果。
type SlideMatchResult struct {
	Target     [2]int  `json:"target"`     // [x, y] 中心坐标
	TargetX    int     `json:"target_x"`   // x 中心坐标
	TargetY    int     `json:"target_y"`   // y 中心坐标
	Confidence float64 `json:"confidence"` // 匹配置信度 (0-1)
}

// SlideMatch 滑块验证码匹配。
//
// targetImage: 滑块图片（字节数据，支持 PNG/JPEG 等格式）
// backgroundImage: 带缺口的背景图片（字节数据）
// simpleTarget: true 使用直接模板匹配；false 使用边缘检测匹配（默认推荐）。
//
// 返回匹配结果，包含滑块目标中心坐标。
func (ocr *DdddOcr) SlideMatch(targetImage, backgroundImage []byte, simpleTarget bool) (*SlideMatchResult, error) {
	target, err := decodeImage(targetImage)
	if err != nil {
		return nil, fmt.Errorf("解码滑块图片失败: %w", err)
	}
	background, err := decodeImage(backgroundImage)
	if err != nil {
		return nil, fmt.Errorf("解码背景图片失败: %w", err)
	}

	targetGray := imageToGray(target)
	backgroundGray := imageToGray(background)

	if simpleTarget {
		return simpleTemplateMatch(targetGray, backgroundGray)
	}
	return edgeBasedMatch(targetGray, backgroundGray)
}

// SlideComparison 滑块缺口对比匹配。
//
// targetImage: 带缺口的图片（字节数据）
// backgroundImage: 完整的背景图片（字节数据）
//
// 通过计算两张图片的差异来定位缺口位置。
func (ocr *DdddOcr) SlideComparison(targetImage, backgroundImage []byte) (*SlideMatchResult, error) {
	img1, err := decodeImage(targetImage)
	if err != nil {
		return nil, fmt.Errorf("解码目标图片失败: %w", err)
	}
	img2, err := decodeImage(backgroundImage)
	if err != nil {
		return nil, fmt.Errorf("解码背景图片失败: %w", err)
	}

	return diffBasedComparison(img1, img2)
}

// simpleTemplateMatch 直接归一化互相关模板匹配。
func simpleTemplateMatch(target, background *image.Gray) (*SlideMatchResult, error) {
	bgW, bgH := background.Bounds().Dx(), background.Bounds().Dy()
	tW, tH := target.Bounds().Dx(), target.Bounds().Dy()

	if tW > bgW || tH > bgH {
		return nil, fmt.Errorf("滑块图片 (%dx%d) 大于背景图片 (%dx%d)", tW, tH, bgW, bgH)
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

// edgeBasedMatch 先做边缘检测再做模板匹配。
func edgeBasedMatch(target, background *image.Gray) (*SlideMatchResult, error) {
	targetEdges := sobelEdges(target)
	backgroundEdges := sobelEdges(background)

	return simpleTemplateMatch(targetEdges, backgroundEdges)
}

// diffBasedComparison 通过图像差异定位缺口。
func diffBasedComparison(img1, img2 image.Image) (*SlideMatchResult, error) {
	bounds1 := img1.Bounds()
	bounds2 := img2.Bounds()

	w := min(bounds1.Dx(), bounds2.Dx())
	h := min(bounds1.Dy(), bounds2.Dy())

	// 计算灰度绝对差异
	diff := image.NewGray(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			g1 := luminance(img1.At(x+bounds1.Min.X, y+bounds1.Min.Y))
			g2 := luminance(img2.At(x+bounds2.Min.X, y+bounds2.Min.Y))
			d := abs(g1 - g2)
			diff.SetGray(x, y, color.Gray{Y: uint8(d)})
		}
	}

	// 二值化，阈值 30
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

	// 形态学闭运算 + 开运算（3x3 核）
	binary = morphologyClose(binary, 3)
	binary = morphologyOpen(binary, 3)

	// 找到最大连通域的中心
	centerX, centerY := findLargestBlobCenter(binary)

	return &SlideMatchResult{
		Target:  [2]int{centerX, centerY},
		TargetX: centerX,
		TargetY: centerY,
	}, nil
}

// ------------------- 图像处理工具函数 -------------------

// float64Slice 提取像素数据为 float64 切片。
func float64Slice(pix []uint8, w, h int) []float64 {
	out := make([]float64, w*h)
	for i := 0; i < w*h && i < len(pix); i++ {
		out[i] = float64(pix[i])
	}
	return out
}

// luminance 将颜色转为灰度亮度值 (0-255)，使用 ITU-R BT.601 标准。
func luminance(c color.Color) int {
	r, g, b, _ := c.RGBA()
	return int(0.299*float64(r>>8) + 0.587*float64(g>>8) + 0.114*float64(b>>8))
}

// nccMatch 归一化互相关模板匹配，返回最佳匹配位置和分数。
func nccMatch(bg []float64, bgW, bgH int, tmpl []float64, tW, tH int) (int, int, float64) {
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

	if bestScore > 1.0 {
		bestScore = 1.0
	}
	if bestScore < 0 {
		bestScore = 0
	}

	return bestX, bestY, bestScore
}

// nccScore 计算指定位置的归一化互相关分数。
func nccScore(bg []float64, bgW, x, y int, tmpl []float64, tW, tH int, tMean, tStd float64) float64 {
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

// sobelEdges 对灰度图进行 Sobel 边缘检测。
func sobelEdges(img *image.Gray) *image.Gray {
	w, h := img.Bounds().Dx(), img.Bounds().Dy()
	edges := image.NewGray(image.Rect(0, 0, w, h))

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

// morphologyClose 形态学闭运算（先膨胀后腐蚀）。
func morphologyClose(img *image.Gray, kernelSize int) *image.Gray {
	return morphologyErode(morphologyDilate(img, kernelSize), kernelSize)
}

// morphologyOpen 形态学开运算（先腐蚀后膨胀）。
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

// findLargestBlobCenter 找到二值图像中最大白色连通域的中心坐标。
func findLargestBlobCenter(img *image.Gray) (int, int) {
	w, h := img.Bounds().Dx(), img.Bounds().Dy()
	visited := make([]bool, w*h)

	bestArea := 0
	bestCX, bestCY := 0, 0

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			if img.GrayAt(x, y).Y > 128 && !visited[y*w+x] {
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

// bfsBlob 广度优先搜索连通区域，返回面积和坐标累加值。
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

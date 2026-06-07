// Package ddddocr 基于 ONNX 模型提供验证码 OCR 识别功能。
//
// 这是 Python ddddocr 库的 Go 移植版本，使用相同的 common.onnx
// 和 common_old.onnx 模型进行图片文字识别。
//
// 基本用法：
//
//	ocr := ddddocr.New(nil)
//	defer ocr.Close()
//	result, err := ocr.Classify(imageBytes)
package ddddocr

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

var (
	initOnce sync.Once
	initErr  error
)

// initializeORT 初始化 ONNX Runtime 环境，可安全重复调用。
func initializeORT(libPath string) {
	initOnce.Do(func() {
		// 将相对路径转为绝对路径再传给 ONNX Runtime C API
		resolvePath := func(p string) string {
			abs, err := filepath.Abs(p)
			if err == nil {
				return abs
			}
			return p
		}

		if libPath != "" {
			ort.SetSharedLibraryPath(resolvePath(libPath))
		} else if envPath := os.Getenv("ONNXRUNTIME_LIB_PATH"); envPath != "" {
			ort.SetSharedLibraryPath(resolvePath(envPath))
		}
		initErr = ort.InitializeEnvironment()
	})
}

// Options 配置引擎的参数。
type Options struct {
	// Old 使用旧版 OCR 模型 common_old.onnx（默认）。
	Old bool
	// Beta 使用新版 OCR 模型 common.onnx，优先级高于 Old。
	Beta bool
	// Det 启用目标检测模式（使用 common_det.onnx）。
	Det bool
	// ModelPath 自定义 ONNX 模型文件路径。设置后 Old/Beta/Det 被忽略。
	ModelPath string
	// CharsetPath 自定义字符集 JSON 文件路径（仅 OCR 模式）。
	CharsetPath string
	// OnnxRuntimeLibPath ONNX Runtime 动态库路径。为空时自动检测。
	OnnxRuntimeLibPath string
}

// DdddOcr 是引擎主体，支持 OCR 识别和目标检测。
type DdddOcr struct {
	options     Options
	charset     []string
	session     *ort.DynamicAdvancedSession // OCR 模型会话
	detSession  *ort.DynamicAdvancedSession // 检测模型会话
	isDet       bool                        // 是否为检测模式
	initialized bool
}

// New 创建一个新的 DdddOcr 实例。
// opts 为 nil 时使用默认配置（旧版模型）。
func New(opts *Options) (*DdddOcr, error) {
	if opts == nil {
		opts = &Options{}
	}

	// 初始化 ONNX Runtime（路径来自 Options 或 ONNXRUNTIME_LIB_PATH 环境变量）
	initializeORT(opts.OnnxRuntimeLibPath)
	if initErr != nil {
		return nil, fmt.Errorf("ONNX Runtime 初始化失败: %w", initErr)
	}

	ocr := &DdddOcr{
		options: *opts,
	}

	// 检测模式
	if opts.Det {
		ocr.isDet = true

		var modelData []byte
		if opts.ModelPath != "" {
			var err error
			modelData, err = os.ReadFile(opts.ModelPath)
			if err != nil {
				return nil, fmt.Errorf("读取模型文件失败: %w", err)
			}
		} else {
			modelData = commonDetOnnxData
		}

		if len(modelData) == 0 {
			return nil, fmt.Errorf("检测模型数据为空")
		}

		session, err := ort.NewDynamicAdvancedSessionWithONNXData(
			modelData,
			[]string{"images"}, // 输入节点名
			[]string{"output"}, // 输出节点名
			nil,
		)
		if err != nil {
			return nil, fmt.Errorf("创建检测会话失败: %w", err)
		}
		ocr.detSession = session
		ocr.initialized = true
		return ocr, nil
	}

	// 确定字符集
	if opts.CharsetPath != "" {
		charsetData, err := os.ReadFile(opts.CharsetPath)
		if err != nil {
			return nil, fmt.Errorf("读取字符集文件失败: %w", err)
		}
		var err2 error
		ocr.charset, err2 = parseCharsetJSON(charsetData)
		if err2 != nil {
			return nil, fmt.Errorf("解析字符集失败: %w", err2)
		}
	} else {
		ocr.charset = getCharset(opts.Old, opts.Beta)
	}

	if len(ocr.charset) == 0 {
		return nil, fmt.Errorf("字符集为空")
	}

	// 加载 ONNX 模型
	var modelData []byte
	if opts.ModelPath != "" {
		var err error
		modelData, err = os.ReadFile(opts.ModelPath)
		if err != nil {
			return nil, fmt.Errorf("读取模型文件失败: %w", err)
		}
	} else {
		modelData = getModelData(opts.Old, opts.Beta)
	}

	if len(modelData) == 0 {
		return nil, fmt.Errorf("模型数据为空")
	}

	// 创建动态会话（支持可变输入输出形状）
	session, err := ort.NewDynamicAdvancedSessionWithONNXData(
		modelData,
		[]string{"input1"}, // 输入节点名
		[]string{"387"},    // 输出节点名
		nil,                // 使用默认会话选项（CPU）
	)
	if err != nil {
		return nil, fmt.Errorf("创建 ONNX 会话失败: %w", err)
	}
	ocr.session = session
	ocr.initialized = true

	return ocr, nil
}

// Classify 对图片进行 OCR 识别。
// 图片支持 PNG、JPEG、BMP、GIF、TIFF 格式。
//
// 返回识别的文字。
func (ocr *DdddOcr) Classify(imgData []byte) (string, error) {
	return ocr.ClassifyWithPNGFix(imgData, false)
}

// ClassifyWithPNGFix 执行 OCR 识别，可选 PNG 透明背景修复。
// pngFix 为 true 时，RGBA 图片会被合成到白色背景上再处理（对应 Python 版的 png_fix）。
func (ocr *DdddOcr) ClassifyWithPNGFix(imgData []byte, pngFix bool) (string, error) {
	if !ocr.initialized {
		return "", fmt.Errorf("OCR 引擎未初始化")
	}

	// 解码图片
	img, err := decodeImage(imgData)
	if err != nil {
		return "", fmt.Errorf("图片解码失败: %w", err)
	}

	// 预处理
	inputData, height, width, err := preprocessWithPNGFix(img, pngFix)
	if err != nil {
		return "", fmt.Errorf("预处理失败: %w", err)
	}

	// 执行推理
	outputData, seqLen, err := ocr.runInference(inputData, height, width)
	if err != nil {
		return "", fmt.Errorf("推理失败: %w", err)
	}

	// CTC 解码
	numClasses := len(ocr.charset)
	result := ctcDecode(outputData, seqLen, numClasses, ocr.charset)

	return result, nil
}

// runInference 创建 ONNX 张量并执行模型推理。
func (ocr *DdddOcr) runInference(inputData []float32, height, width int) ([]float32, int, error) {
	// 创建输入张量 (1, 1, 64, width)
	inputShape := ort.NewShape(1, 1, int64(height), int64(width))
	inputTensor, err := ort.NewTensor(inputShape, inputData)
	if err != nil {
		return nil, 0, fmt.Errorf("创建输入张量失败: %w", err)
	}
	defer inputTensor.Destroy()

	// 创建输出张量，序列长度 = ceil(width / 8)
	seqLen := int64((width + 7) / 8)
	if seqLen < 1 {
		seqLen = 1
	}
	numClasses := int64(len(ocr.charset))

	outputShape := ort.NewShape(seqLen, 1, numClasses)
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		return nil, 0, fmt.Errorf("创建输出张量失败: %w", err)
	}
	defer outputTensor.Destroy()

	// 执行推理
	err = ocr.session.Run(
		[]ort.Value{inputTensor},
		[]ort.Value{outputTensor},
	)
	if err != nil {
		return nil, 0, fmt.Errorf("ONNX 推理失败: %w", err)
	}

	// 获取输出数据
	outputData := outputTensor.GetData()

	return outputData, int(seqLen), nil
}

// Close 释放引擎占用的所有资源。
func (ocr *DdddOcr) Close() error {
	if ocr.session != nil {
		ocr.session.Destroy()
		ocr.session = nil
	}
	if ocr.detSession != nil {
		ocr.detSession.Destroy()
		ocr.detSession = nil
	}
	ocr.initialized = false
	return nil
}

// GetCharset 返回当前 OCR 实例使用的字符集。
func (ocr *DdddOcr) GetCharset() []string {
	result := make([]string, len(ocr.charset))
	copy(result, ocr.charset)
	return result
}

// IsInitialized 返回 OCR 引擎是否已就绪。
func (ocr *DdddOcr) IsInitialized() bool {
	return ocr.initialized
}

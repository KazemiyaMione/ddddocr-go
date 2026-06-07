// Package ddddocr provides OCR-based captcha recognition using ONNX models.
//
// It is a Go port of the Python ddddocr library, using the same common.onnx
// and common_old.onnx models for text recognition from images.
//
// Basic usage:
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

// initializeORT initializes ONNX Runtime. Safe to call multiple times.
func initializeORT(libPath string) {
	initOnce.Do(func() {
		// Resolve relative path to absolute before passing to ONNX Runtime C API
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

// Options configures the OCR engine.
type Options struct {
	// Old uses the old model (common_old.onnx). This is the default if Beta is also false.
	Old bool
	// Beta uses the beta/new model (common.onnx). Takes precedence over Old.
	Beta bool
	// ModelPath specifies a custom ONNX model file path. If set, Old/Beta are ignored.
	ModelPath string
	// CharsetPath specifies a custom charset JSON file path.
	CharsetPath string
	// OnnxRuntimeLibPath specifies the path to the ONNX Runtime shared library.
	// If empty, auto-detection is attempted.
	OnnxRuntimeLibPath string
}

// DdddOcr is the main OCR engine.
type DdddOcr struct {
	options     Options
	charset     []string
	session     *ort.DynamicAdvancedSession
	initialized bool
}

// New creates a new DdddOcr instance with the given options.
// If opts is nil, defaults are used (old model).
func New(opts *Options) (*DdddOcr, error) {
	if opts == nil {
		opts = &Options{}
	}

	// Initialize ONNX Runtime (path from Options or ONNXRUNTIME_LIB_PATH env var)
	initializeORT(opts.OnnxRuntimeLibPath)
	if initErr != nil {
		return nil, fmt.Errorf("failed to initialize ONNX Runtime: %w", initErr)
	}

	ocr := &DdddOcr{
		options: *opts,
	}

	// Determine charset
	if opts.CharsetPath != "" {
		charsetData, err := os.ReadFile(opts.CharsetPath)
		if err != nil {
			return nil, fmt.Errorf("failed to read charset file: %w", err)
		}
		var err2 error
		ocr.charset, err2 = parseCharsetJSON(charsetData)
		if err2 != nil {
			return nil, fmt.Errorf("failed to parse charset: %w", err2)
		}
	} else {
		ocr.charset = getCharset(opts.Old, opts.Beta)
	}

	if len(ocr.charset) == 0 {
		return nil, fmt.Errorf("charset is empty")
	}

	// Load ONNX model
	var modelData []byte
	if opts.ModelPath != "" {
		var err error
		modelData, err = os.ReadFile(opts.ModelPath)
		if err != nil {
			return nil, fmt.Errorf("failed to read model file: %w", err)
		}
	} else {
		modelData = getModelData(opts.Old, opts.Beta)
	}

	if len(modelData) == 0 {
		return nil, fmt.Errorf("model data is empty")
	}

	// Create dynamic session (supports variable input/output shapes)
	session, err := ort.NewDynamicAdvancedSessionWithONNXData(
		modelData,
		[]string{"input1"}, // input names
		[]string{"387"},    // output names
		nil,                // use default session options (CPU)
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create ONNX session: %w", err)
	}
	ocr.session = session
	ocr.initialized = true

	return ocr, nil
}

// Classify performs OCR on the given image data.
// The image can be in PNG, JPEG, BMP, GIF, or TIFF format.
//
// Returns the recognized text.
func (ocr *DdddOcr) Classify(imgData []byte) (string, error) {
	return ocr.ClassifyWithPNGFix(imgData, false)
}

// ClassifyWithPNGFix performs OCR with optional PNG transparent background fix.
// When pngFix is true, RGBA images are composited over a white background
// before processing (matching Python's png_fix behavior).
func (ocr *DdddOcr) ClassifyWithPNGFix(imgData []byte, pngFix bool) (string, error) {
	if !ocr.initialized {
		return "", fmt.Errorf("OCR engine not initialized")
	}

	// Decode image
	img, err := decodeImage(imgData)
	if err != nil {
		return "", fmt.Errorf("failed to decode image: %w", err)
	}

	// Preprocess
	inputData, height, width, err := preprocessWithPNGFix(img, pngFix)
	if err != nil {
		return "", fmt.Errorf("preprocessing failed: %w", err)
	}

	// Run inference
	outputData, seqLen, err := ocr.runInference(inputData, height, width)
	if err != nil {
		return "", fmt.Errorf("inference failed: %w", err)
	}

	// CTC decode
	numClasses := len(ocr.charset)
	result := ctcDecode(outputData, seqLen, numClasses, ocr.charset)

	return result, nil
}

// runInference creates ONNX tensors and runs the model.
func (ocr *DdddOcr) runInference(inputData []float32, height, width int) ([]float32, int, error) {
	// Create input tensor (1, 1, 64, width)
	inputShape := ort.NewShape(1, 1, int64(height), int64(width))
	inputTensor, err := ort.NewTensor(inputShape, inputData)
	if err != nil {
		return nil, 0, fmt.Errorf("failed to create input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	// Create output tensor with exact sequence length
	// seqLen = ceil(width / 8) — verified against actual model output
	seqLen := int64((width + 7) / 8)
	if seqLen < 1 {
		seqLen = 1
	}
	numClasses := int64(len(ocr.charset))

	outputShape := ort.NewShape(seqLen, 1, numClasses)
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		return nil, 0, fmt.Errorf("failed to create output tensor: %w", err)
	}
	defer outputTensor.Destroy()

	// Run inference
	err = ocr.session.Run(
		[]ort.Value{inputTensor},
		[]ort.Value{outputTensor},
	)
	if err != nil {
		return nil, 0, fmt.Errorf("ONNX inference failed: %w", err)
	}

	// Get output data
	outputData := outputTensor.GetData()

	return outputData, int(seqLen), nil
}

// Close releases all resources used by the OCR engine.
func (ocr *DdddOcr) Close() error {
	if ocr.session != nil {
		ocr.session.Destroy()
		ocr.session = nil
	}
	ocr.initialized = false
	return nil
}

// GetCharset returns the character set used by this OCR instance.
func (ocr *DdddOcr) GetCharset() []string {
	result := make([]string, len(ocr.charset))
	copy(result, ocr.charset)
	return result
}

// IsInitialized returns whether the OCR engine is ready.
func (ocr *DdddOcr) IsInitialized() bool {
	return ocr.initialized
}

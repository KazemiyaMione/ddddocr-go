# ddddocr-go

Go port of [ddddocr](https://github.com/sml2h3/ddddocr) — captcha OCR recognition using ONNX models.

Uses the same `common.onnx` and `common_old.onnx` models from the Python library for text recognition from captcha images.

## Requirements

- **Go 1.21+**
- **ONNX Runtime** shared library installed on your system
  - **Windows**: Download `onnxruntime-win-x64-1.x.x.zip` from [ONNX Runtime releases](https://github.com/microsoft/onnxruntime/releases), extract `onnxruntime.dll`
  - **Linux**: `libonnxruntime.so` (install via package manager or download)
  - **macOS**: `libonnxruntime.dylib`

## Installation

```bash
go get github.com/bronya/ddddocr-go
```

## Quick Start

```go
package main

import (
    "fmt"
    "os"

    "github.com/bronya/ddddocr-go"
)

func main() {
    // Read image
    imgData, _ := os.ReadFile("captcha.png")

    // Create OCR engine (uses old model by default)
    ocr, err := ddddocr.New(nil)
    if err != nil {
        panic(err)
    }
    defer ocr.Close()

    // Recognize
    result, _ := ocr.Classify(imgData)
    fmt.Println(result)
}
```

## CLI Tool

```bash
# Build CLI
go install github.com/bronya/ddddocr-go/cmd/ddddocr@latest

# Use default (old) model
ddddocr captcha.png

# Use beta model
ddddocr --beta captcha.png

# With PNG transparent background fix
ddddocr --pngfix captcha.png
```

## API

### Options

```go
opts := &ddddocr.Options{
    Old:                false,  // Use old model (common_old.onnx)
    Beta:               true,   // Use beta model (common.onnx)
    ModelPath:          "",     // Custom ONNX model path
    CharsetPath:        "",     // Custom charset JSON path
    OnnxRuntimeLibPath: "",     // Override ONNX Runtime library path
}
```

### Methods

- `New(opts *Options) (*DdddOcr, error)` — Create OCR engine
- `Classify(imgData []byte) (string, error)` — Recognize text from image bytes
- `ClassifyWithPNGFix(imgData []byte, pngFix bool) (string, error)` — With PNG alpha fix
- `Close() error` — Release resources
- `GetCharset() []string` — Get the character set
- `IsInitialized() bool` — Check if ready

## Models

| Model | File | Charset | Description |
|-------|------|---------|-------------|
| Default (Old) | `common_old.onnx` | CHARSET_OLD (8210 chars) | Standard captcha recognition |
| Beta | `common.onnx` | CHARSET_BETA (8210 chars) | Newer trained model |

Both models accept:
- **Input**: `(1, 1, 64, image_width)` grayscale float32 [0,1]
- **Output**: `(seqlen, 1, 8210)` CTC logits

## Image Preprocessing

1. Decode image (PNG, JPEG, BMP, GIF, TIFF)
2. Convert to grayscale
3. Resize: height=64px, width scaled proportionally (bilinear)
4. Normalize: divide by 255.0
5. Reshape to NCHW format `(1, 1, 64, width)`

## CTC Decoding

1. argmax per timestep along class dimension
2. Remove consecutive duplicate predictions
3. Skip blank label (index 0)
4. Map remaining indices to character set characters

## License

Same as the original ddddocr project.

## Credits

Based on [sml2h3/ddddocr](https://github.com/sml2h3/ddddocr) — the original Python implementation.

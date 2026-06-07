# ddddocr-go

[ddddocr](https://github.com/sml2h3/ddddocr) 的 Go 语言移植版，基于 ONNX 模型实现验证码识别。

## 功能

| 功能 | 方法 | 模型 |
|------|------|------|
| OCR 文字识别 | `Classify()` | `common_old.onnx` / `common.onnx` |
| 目标检测 | `Detection()` | `common_det.onnx` |
| 滑块匹配 | `SlideMatch()` / `SlideComparison()` | 纯算法，无需模型 |

## 安装

### 1. 安装 ONNX Runtime

从 [ONNX Runtime Releases](https://github.com/microsoft/onnxruntime/releases/tag/v1.25.0) 下载对应系统的动态库：

- Windows: `onnxruntime-win-x64-1.25.0.zip` → 解压得到 `onnxruntime.dll`
- Linux: `onnxruntime-linux-x64-1.25.0.tgz` → `libonnxruntime.so`
- macOS: `onnxruntime-osx-universal2-1.25.0.tgz` → `libonnxruntime.dylib`

### 2. 引入库

```bash
go get github.com/KazemiyaMione/ddddocr-go@latest
```

## 快速开始

```go
package main

import (
    "fmt"
    "os"
    "github.com/KazemiyaMione/ddddocr-go"
)

func main() {
    opts := &ddddocr.Options{
        OnnxRuntimeLibPath: "./onnxruntime.dll",
    }
    ocr, err := ddddocr.New(opts)
    if err != nil {
        panic(err)
    }
    defer ocr.Close()

    data, _ := os.ReadFile("captcha.png")
    result, _ := ocr.Classify(data)
    fmt.Println(result)
}
```

## OCR 文字识别

```go
// 默认旧版模型
ocr, _ := ddddocr.New(nil)

// 新版模型
ocr, _ := ddddocr.New(&ddddocr.Options{Beta: true})

// 识别
result, _ := ocr.Classify(imageBytes)

// PNG 透明背景修复
result, _ := ocr.ClassifyWithPNGFix(imageBytes, true)
```

## 目标检测

```go
ocr, _ := ddddocr.New(&ddddocr.Options{Det: true})

boxes, _ := ocr.Detection(imageBytes)
for _, b := range boxes {
    // b = [x1, y1, x2, y2] 左上角和右下角坐标
    fmt.Printf("检测到目标: (%d,%d) -> (%d,%d)\n", b[0], b[1], b[2], b[3])
}
```

## 滑块匹配

```go
ocr, _ := ddddocr.New(nil)

// 边缘检测匹配（推荐）
result, _ := ocr.SlideMatch(sliderBytes, backgroundBytes, false)

// 简单模板匹配
result, _ := ocr.SlideMatch(sliderBytes, backgroundBytes, true)

// 缺口对比
result, _ := ocr.SlideComparison(gapImageBytes, fullImageBytes)

fmt.Println(result.TargetX, result.TargetY) // 滑块中心坐标
fmt.Println(result.Confidence)               // 置信度 0-1
```

## Options 参数

```go
type Options struct {
    Old    bool   // 旧版 OCR 模型（默认）
    Beta   bool   // 新版 OCR 模型
    Det    bool   // 目标检测模式
    ModelPath    string // 自定义 ONNX 模型路径
    CharsetPath  string // 自定义字符集 JSON 路径（仅 OCR）
    OnnxRuntimeLibPath string // ONNX Runtime 动态库路径
}
```

## 环境变量

```bash
# 设置 ONNX Runtime 库路径（优先级低于 Options.OnnxRuntimeLibPath）
export ONNXRUNTIME_LIB_PATH=/path/to/onnxruntime.dll
```

## 与 Python 版对比

| 测试项 | Python | Go |
|--------|--------|-----|
| OCR 识别 | ✓ | ✓ 完全一致 |
| 目标检测 (PNG) | ✓ | ✓ 1px 误差 |
| 目标检测 (JPG) | ✓ | 建议转 PNG |
| 滑块匹配 | ✓ | ✓ |

## 许可

MIT License（与原始 ddddocr 项目一致）

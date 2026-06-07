// Command ddddocr 是基于 ONNX 模型的验证码 OCR 命令行工具。
//
// 用法：
//
//	ddddocr <图片路径>
//	ddddocr --beta <图片路径>
//	ddddocr --old <图片路径>
//	ddddocr --model <模型路径> --charset <字符集路径> <图片路径>
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/KazemiyaMione/ddddocr-go"
)

func main() {
	var (
		beta    bool
		old     bool
		model   string
		charset string
		pngFix  bool
		libPath string
	)

	flag.BoolVar(&beta, "beta", false, "使用新版模型 common.onnx")
	flag.BoolVar(&old, "old", false, "使用旧版模型 common_old.onnx（默认）")
	flag.StringVar(&model, "model", "", "自定义 ONNX 模型文件路径")
	flag.StringVar(&charset, "charset", "", "自定义字符集 JSON 文件路径")
	flag.BoolVar(&pngFix, "pngfix", false, "修复 PNG 透明背景（合成到白色背景上）")
	flag.StringVar(&libPath, "lib", "", "ONNX Runtime 动态库路径")
	flag.Parse()

	args := flag.Args()
	if len(args) < 1 {
		fmt.Fprintf(os.Stderr, "用法: ddddocr [参数] <图片路径>\n")
		flag.PrintDefaults()
		os.Exit(1)
	}

	imagePath := args[0]

	// 读取图片文件
	imgData, err := os.ReadFile(imagePath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "读取图片文件失败: %v\n", err)
		os.Exit(1)
	}

	// 创建 OCR 引擎
	opts := &ddddocr.Options{
		Old:                old,
		Beta:               beta,
		ModelPath:          model,
		CharsetPath:        charset,
		OnnxRuntimeLibPath: libPath,
	}
	ocr, err := ddddocr.New(opts)
	if err != nil {
		fmt.Fprintf(os.Stderr, "初始化 OCR 引擎失败: %v\n", err)
		os.Exit(1)
	}
	defer ocr.Close()

	// 执行识别
	result, err := ocr.ClassifyWithPNGFix(imgData, pngFix)
	if err != nil {
		fmt.Fprintf(os.Stderr, "识别失败: %v\n", err)
		os.Exit(1)
	}

	fmt.Println(result)
}

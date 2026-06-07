package main

import (
	"fmt"
	"os"

	"github.com/KazemiyaMione/ddddocr-go"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "用法: basic_ocr <图片路径>\n")
		os.Exit(1)
	}

	imagePath := os.Args[1]

	// 读取图片文件
	imgData, err := os.ReadFile(imagePath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "读取图片失败: %v\n", err)
		os.Exit(1)
	}

	// 创建 OCR 引擎（默认使用旧版模型）
	ocr, err := ddddocr.New(nil)
	if err != nil {
		fmt.Fprintf(os.Stderr, "创建 OCR 引擎失败: %v\n", err)
		os.Exit(1)
	}
	defer ocr.Close()

	// 识别文字
	result, err := ocr.Classify(imgData)
	if err != nil {
		fmt.Fprintf(os.Stderr, "识别失败: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("结果:", result)
}

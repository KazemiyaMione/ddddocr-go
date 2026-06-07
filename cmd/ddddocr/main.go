// Command ddddocr is a CLI tool for OCR-based captcha recognition.
//
// Usage:
//
//	ddddocr <image_path>
//	ddddocr --beta <image_path>
//	ddddocr --old <image_path>
//	ddddocr --model <model_path> --charset <charset_path> <image_path>
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

	flag.BoolVar(&beta, "beta", false, "Use beta/new model (common.onnx)")
	flag.BoolVar(&old, "old", false, "Use old model (common_old.onnx, default)")
	flag.StringVar(&model, "model", "", "Path to custom ONNX model")
	flag.StringVar(&charset, "charset", "", "Path to custom charset JSON file")
	flag.BoolVar(&pngFix, "pngfix", false, "Fix PNG transparent background (composite over white)")
	flag.StringVar(&libPath, "lib", "", "Path to ONNX Runtime shared library")
	flag.Parse()

	args := flag.Args()
	if len(args) < 1 {
		fmt.Fprintf(os.Stderr, "Usage: ddddocr [flags] <image_path>\n")
		flag.PrintDefaults()
		os.Exit(1)
	}

	imagePath := args[0]

	// Read image file
	imgData, err := os.ReadFile(imagePath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error reading image file: %v\n", err)
		os.Exit(1)
	}

	// Create OCR engine
	opts := &ddddocr.Options{
		Old:                old,
		Beta:               beta,
		ModelPath:          model,
		CharsetPath:        charset,
		OnnxRuntimeLibPath: libPath,
	}
	ocr, err := ddddocr.New(opts)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error initializing OCR engine: %v\n", err)
		os.Exit(1)
	}
	defer ocr.Close()

	// Run recognition
	result, err := ocr.ClassifyWithPNGFix(imgData, pngFix)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error during OCR: %v\n", err)
		os.Exit(1)
	}

	fmt.Println(result)
}

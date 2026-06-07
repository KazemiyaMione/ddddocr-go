package main

import (
	"fmt"
	"os"

	"github.com/bronya/ddddocr-go"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "Usage: basic_ocr <image_path>\n")
		os.Exit(1)
	}

	imagePath := os.Args[1]

	// Read image file
	imgData, err := os.ReadFile(imagePath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error reading image: %v\n", err)
		os.Exit(1)
	}

	// Create OCR engine (default: old model)
	ocr, err := ddddocr.New(nil)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating OCR engine: %v\n", err)
		os.Exit(1)
	}
	defer ocr.Close()

	// Recognize text
	result, err := ocr.Classify(imgData)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error recognizing text: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("Result:", result)
}

package ddddocr

import (
	"encoding/json"
	_ "embed"
	"fmt"
)

//go:embed models/common.onnx
var commonOnnxData []byte

//go:embed models/common_old.onnx
var commonOldOnnxData []byte

//go:embed models/charset_old.json
var charsetOldJSON []byte

//go:embed models/charset_beta.json
var charsetBetaJSON []byte

// charsetCache holds parsed charsets
var charsetCache = map[string][]string{}

func init() {
	// Parse charsets at init time
	old, err := parseCharsetJSON(charsetOldJSON)
	if err != nil {
		panic(fmt.Sprintf("failed to parse charset_old.json: %v", err))
	}
	charsetCache["old"] = old

	beta, err := parseCharsetJSON(charsetBetaJSON)
	if err != nil {
		panic(fmt.Sprintf("failed to parse charset_beta.json: %v", err))
	}
	charsetCache["beta"] = beta
}

func parseCharsetJSON(data []byte) ([]string, error) {
	var charset []string
	if err := json.Unmarshal(data, &charset); err != nil {
		return nil, err
	}
	return charset, nil
}

// getCharset returns the charset for the given model type
func getCharset(old bool, beta bool) []string {
	if beta {
		return charsetCache["beta"]
	}
	return charsetCache["old"]
}

// getModelData returns the ONNX model data for the given model type
func getModelData(old bool, beta bool) []byte {
	if beta {
		return commonOnnxData
	}
	return commonOldOnnxData
}

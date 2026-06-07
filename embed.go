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

// charsetCache 缓存已解析的字符集。
var charsetCache = map[string][]string{}

func init() {
	old, err := parseCharsetJSON(charsetOldJSON)
	if err != nil {
		panic(fmt.Sprintf("解析 charset_old.json 失败: %v", err))
	}
	charsetCache["old"] = old

	beta, err := parseCharsetJSON(charsetBetaJSON)
	if err != nil {
		panic(fmt.Sprintf("解析 charset_beta.json 失败: %v", err))
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

// getCharset 返回指定模型类型对应的字符集。
func getCharset(old bool, beta bool) []string {
	if beta {
		return charsetCache["beta"]
	}
	return charsetCache["old"]
}

// getModelData 返回指定模型类型对应的 ONNX 模型数据。
func getModelData(old bool, beta bool) []byte {
	if beta {
		return commonOnnxData
	}
	return commonOldOnnxData
}

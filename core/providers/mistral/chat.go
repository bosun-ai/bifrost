package mistral

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"

	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

// handleChatResponse separates native Mistral thinking from visible text before
// the shared schema can discard it. The same conversion handles SSE deltas.
func handleChatResponse(body []byte, response *schemas.BifrostChatResponse, request []byte, rawRequest, rawResponse bool) (interface{}, interface{}, *schemas.BifrostError) {
	normalized, err := normalizeChatResponse(body)
	if err != nil {
		return nil, nil, providerUtils.NewBifrostOperationError(schemas.ErrProviderResponseUnmarshal, err)
	}
	req, _, bifrostErr := providerUtils.HandleProviderResponse(normalized, response, request, rawRequest, false)
	if !rawResponse {
		return req, nil, bifrostErr
	}
	var compact bytes.Buffer
	if err := json.Compact(&compact, body); err != nil {
		return req, nil, bifrostErr
	}
	return req, json.RawMessage(compact.Bytes()), bifrostErr
}

func normalizeChatResponse(body []byte) ([]byte, error) {
	if !gjson.ValidBytes(body) {
		return nil, fmt.Errorf("invalid Mistral response JSON")
	}
	result := body
	for i, choice := range gjson.GetBytes(body, "choices").Array() {
		for _, field := range []string{"message", "delta"} {
			content := choice.Get(field + ".content")
			if !content.IsArray() {
				continue
			}
			var text, reasoning strings.Builder
			hasText, hasThinking := false, false
			for _, block := range content.Array() {
				switch block.Get("type").String() {
				case "text":
					if block.Get("text").Type != gjson.String {
						return nil, fmt.Errorf("invalid Mistral text block")
					}
					hasText = true
					text.WriteString(block.Get("text").String())
				case "thinking":
					thinking := block.Get("thinking")
					if !thinking.IsArray() {
						return nil, fmt.Errorf("invalid Mistral thinking block")
					}
					hasThinking = true
					for _, part := range thinking.Array() {
						if part.Get("type").String() != "text" || part.Get("text").Type != gjson.String {
							return nil, fmt.Errorf("unsupported Mistral thinking content")
						}
						reasoning.WriteString(part.Get("text").String())
					}
				default:
					return nil, fmt.Errorf("unsupported Mistral response content block")
				}
			}
			prefix := fmt.Sprintf("choices.%d.%s", i, field)
			var visible interface{}
			if hasText {
				visible = text.String()
			}
			var err error
			result, err = sjson.SetBytes(result, prefix+".content", visible)
			if err != nil {
				return nil, err
			}
			if hasThinking {
				result, err = sjson.SetBytes(result, prefix+".reasoning_content", reasoning.String())
				if err != nil {
					return nil, err
				}
			}
		}
	}
	return result, nil
}

package openai

import (
	"encoding/json"
	"fmt"

	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

// encodeMistralReasoning replays assistant reasoning using Mistral's native
// content blocks instead of the unsupported OpenAI-compatible reasoning field.
func encodeMistralReasoning(body []byte, messages []OpenAIMessage) ([]byte, error) {
	result := body
	for i, message := range messages {
		if message.Role != schemas.ChatMessageRoleAssistant || message.OpenAIChatAssistantMessage == nil || message.Reasoning == nil {
			continue
		}
		prefix := fmt.Sprintf("messages.%d", i)
		encoded, err := providerUtils.MarshalSorted(map[string]any{
			"type": "thinking", "closed": true,
			"thinking": []map[string]string{{"type": "text", "text": *message.Reasoning}},
		})
		if err != nil {
			return nil, err
		}
		blocks := []json.RawMessage{encoded}
		content := providerUtils.GetJSONField(body, prefix+".content")
		if content.Type == gjson.String {
			text, err := providerUtils.MarshalSorted(map[string]string{"type": "text", "text": content.String()})
			if err != nil {
				return nil, err
			}
			blocks = append(blocks, text)
		} else if content.IsArray() {
			for _, block := range content.Array() {
				blocks = append(blocks, json.RawMessage(block.Raw))
			}
		} else if content.Type != gjson.Null {
			return nil, fmt.Errorf("unsupported Mistral assistant content")
		}
		encoded, err = providerUtils.MarshalSorted(blocks)
		if err != nil {
			return nil, err
		}
		result, err = sjson.SetRawBytes(result, prefix+".content", encoded)
		if err != nil {
			return nil, err
		}
		result, err = sjson.DeleteBytes(result, prefix+".reasoning_content")
		if err != nil {
			return nil, err
		}
	}
	return result, nil
}

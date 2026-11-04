package openai

import (
	"encoding/json"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/stretchr/testify/require"
	"testing"
)

func TestMistralReplaysReasoningAsNativeContent(t *testing.T) {
	input := &schemas.BifrostChatRequest{Provider: schemas.Mistral, Model: "mistral-large-4", Input: []schemas.ChatMessage{{Role: schemas.ChatMessageRoleAssistant, Content: &schemas.ChatMessageContent{ContentStr: schemas.Ptr("visible")}, ChatAssistantMessage: &schemas.ChatAssistantMessage{Reasoning: schemas.Ptr("synthetic thought")}}}}
	request := ToOpenAIChatRequest(nil, input)
	encoded, err := json.Marshal(request)
	require.NoError(t, err)
	var wire map[string]any
	require.NoError(t, json.Unmarshal(encoded, &wire))
	message := wire["messages"].([]any)[0].(map[string]any)
	require.NotContains(t, message, "reasoning_content")
	require.Equal(t, []any{
		map[string]any{"type": "thinking", "thinking": []any{map[string]any{"type": "text", "text": "synthetic thought"}}, "closed": true},
		map[string]any{"type": "text", "text": "visible"},
	}, message["content"])
	require.Equal(t, "visible", *input.Input[0].Content.ContentStr)
	require.Equal(t, "synthetic thought", *input.Input[0].Reasoning)
}

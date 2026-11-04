package mistral

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/maximhq/bifrost/core/schemas"
	"github.com/stretchr/testify/require"
)

func TestMistralThinkingAtHTTPBoundary(t *testing.T) {
	for _, streaming := range []bool{false, true} {
		t.Run(fmt.Sprint(streaming), func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				message := `{"role":"assistant","content":[{"type":"thinking","thinking":[{"type":"text","text":"synthetic thought"}],"closed":true},{"type":"text","text":"visible"}],"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"echo","arguments":"{}"}}]}`
				field, object := "message", "chat.completion"
				if streaming {
					field, object = "delta", "chat.completion.chunk"
				}
				body := fmt.Sprintf(`{"id":"test","object":%q,"created":1,"model":"mistral-large-4","choices":[{"index":0,%q:%s,"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":12,"completion_tokens":3,"total_tokens":15,"prompt_tokens_details":{"cached_tokens":8}}}`, object, field, message)
				if streaming {
					w.Header().Set("Content-Type", "text/event-stream")
					fmt.Fprintf(w, "data: %s\n\ndata: [DONE]\n\n", body)
				} else {
					w.Header().Set("Content-Type", "application/json")
					io.WriteString(w, body)
				}
			}))
			defer server.Close()
			provider := NewMistralProvider(&schemas.ProviderConfig{NetworkConfig: schemas.NetworkConfig{BaseURL: server.URL}}, &testLogger{})
			ctx, cancel := schemas.NewBifrostContextWithCancel(context.Background())
			defer cancel()
			request := &schemas.BifrostChatRequest{Provider: schemas.Mistral, Model: "mistral-large-4", Input: []schemas.ChatMessage{{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: schemas.Ptr("hello")}}}}
			check := func(response *schemas.BifrostChatResponse) {
				encoded, err := json.Marshal(response)
				require.NoError(t, err)
				var wire map[string]any
				require.NoError(t, json.Unmarshal(encoded, &wire))
				choice := wire["choices"].([]any)[0].(map[string]any)
				field := "message"
				if streaming {
					field = "delta"
				}
				message := choice[field].(map[string]any)
				require.Equal(t, "visible", message["content"])
				require.Equal(t, "synthetic thought", message["reasoning_content"])
				require.Len(t, message["tool_calls"], 1)
				if !streaming {
					require.EqualValues(t, 12, wire["usage"].(map[string]any)["prompt_tokens"])
				}
			}
			if streaming {
				hook := func(_ *schemas.BifrostContext, r *schemas.BifrostResponse, e *schemas.BifrostError) (*schemas.BifrostResponse, *schemas.BifrostError) {
					return r, e
				}
				stream, err := provider.ChatCompletionStream(ctx, hook, nil, schemas.Key{}, request)
				require.Nil(t, err)
				count := 0
				for chunk := range stream {
					require.Nil(t, chunk.BifrostError)
					if chunk.BifrostChatResponse != nil && len(chunk.BifrostChatResponse.Choices) > 0 && chunk.BifrostChatResponse.Choices[0].Delta.Content != nil {
						check(chunk.BifrostChatResponse)
						count++
					}
				}
				require.Equal(t, 1, count)
			} else {
				response, err := provider.ChatCompletion(ctx, schemas.Key{}, request)
				require.Nil(t, err)
				check(response)
			}
		})
	}
}

func TestMistralContentNormalization(t *testing.T) {
	for _, tc := range []struct {
		name, content, text, reasoning string
		fails                          bool
	}{
		{"string", `"answer"`, "answer", "", false},
		{"null", `null`, "", "", false},
		{"empty", `[]`, "", "", false},
		{"thinking", `[{"type":"thinking","thinking":[{"type":"text","text":"a"},{"type":"text","text":"b"}],"closed":false}]`, "", "ab", false},
		{"transition", `[{"type":"thinking","thinking":[],"closed":true},{"type":"text","text":"a"},{"type":"text","text":"b"}]`, "ab", "", false},
		{"missing thinking", `[{"type":"thinking"}]`, "", "", true},
		{"unknown", `[{"type":"unknown"}]`, "", "", true},
		{"invalid text", `[{"type":"text","text":7}]`, "", "", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for _, field := range []string{"message", "delta"} {
				body := []byte(fmt.Sprintf(`{"choices":[{"%s":{"content":%s}}],"usage":{"prompt_tokens":12}}`, field, tc.content))
				normalized, err := normalizeChatResponse(body)
				if tc.fails {
					require.Error(t, err)
					continue
				}
				require.NoError(t, err)
				var decoded map[string]any
				require.NoError(t, json.Unmarshal(normalized, &decoded))
				message := decoded["choices"].([]any)[0].(map[string]any)[field].(map[string]any)
				_, isArray := message["content"].([]any)
				require.False(t, isArray)
				if tc.text != "" {
					require.Equal(t, tc.text, message["content"])
				}
				if tc.reasoning != "" {
					require.Equal(t, tc.reasoning, message["reasoning_content"])
				}
				require.EqualValues(t, 12, decoded["usage"].(map[string]any)["prompt_tokens"])
			}
		})
	}
	_, err := normalizeChatResponse([]byte(`{"choices":`))
	require.Error(t, err)
}

func TestMistralRawResponseStaysNative(t *testing.T) {
	body := []byte(`{"choices":[{"message":{"role":"assistant","content":[{"type":"thinking","thinking":[{"type":"text","text":"synthetic"}]}]}}]}`)
	var response schemas.BifrostChatResponse
	_, raw, err := handleChatResponse(body, &response, nil, false, true)
	require.Nil(t, err)
	require.JSONEq(t, string(body), string(raw.(json.RawMessage)))
	require.Equal(t, "synthetic", *response.Choices[0].Message.Reasoning)
}

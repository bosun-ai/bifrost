package bedrock_test

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/maximhq/bifrost/core/providers/bedrock"
	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Converse's ToolResultContentBlock union has no cachePoint member. A cachePoint nested inside
// toolResult.content is accepted over HTTP but produces zero cache reads and writes, so an agent
// loop whose newest message is a tool result never caches. The checkpoint has to be a standalone
// block in the message's outer content array, right after the toolResult it closes over:
//
//	{"role":"user","content":[{"toolResult":{...}},{"cachePoint":{"type":"default"}}]}

const toolResultCacheModel = "eu.anthropic.claude-sonnet-5-5"

// agentLoopHistory is the shape every agent turn ends with: the model called a tool and the
// newest message is that tool's result.
func agentLoopHistory(toolContent schemas.ChatMessageContent) []schemas.ChatMessage {
	return []schemas.ChatMessage{
		{Role: schemas.ChatMessageRoleSystem, Content: &schemas.ChatMessageContent{ContentStr: new("You are a coding agent.")}},
		{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: new("List the files.")}},
		{
			Role: schemas.ChatMessageRoleAssistant,
			ChatAssistantMessage: &schemas.ChatAssistantMessage{
				ToolCalls: []schemas.ChatAssistantMessageToolCall{{
					Type:     new("function"),
					ID:       new("call_1"),
					Function: schemas.ChatAssistantMessageToolCallFunction{Name: new("ls"), Arguments: `{}`},
				}},
			},
		},
		{
			Role:            schemas.ChatMessageRoleTool,
			Content:         &toolContent,
			ChatToolMessage: &schemas.ChatToolMessage{ToolCallID: new("call_1")},
		},
	}
}

func cachedToolText(text string) schemas.ChatMessageContent {
	return schemas.ChatMessageContent{ContentBlocks: []schemas.ChatContentBlock{{
		Type:         schemas.ChatContentBlockTypeText,
		Text:         new(text),
		CacheControl: &schemas.CacheControl{Type: schemas.CacheControlTypeEphemeral},
	}}}
}

func convertChatHistory(t *testing.T, input []schemas.ChatMessage) *bedrock.BedrockConverseRequest {
	t.Helper()
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	req, err := bedrock.ToBedrockChatCompletionRequest(ctx, &schemas.BifrostChatRequest{
		Provider: schemas.Bedrock,
		Model:    toolResultCacheModel,
		Input:    input,
	})
	require.NoError(t, err)
	return req
}

func lastConverseMessageJSON(t *testing.T, req *bedrock.BedrockConverseRequest) string {
	t.Helper()
	require.NotEmpty(t, req.Messages)
	raw, err := json.Marshal(req.Messages[len(req.Messages)-1])
	require.NoError(t, err)
	return string(raw)
}

// toolResultOf returns the only toolResult in a message and asserts no cachePoint hides inside it.
func toolResultOf(t *testing.T, msg bedrock.BedrockMessage) *bedrock.BedrockToolResult {
	t.Helper()
	require.NotEmpty(t, msg.Content)
	result := msg.Content[0].ToolResult
	require.NotNil(t, result, "tool result must lead the message")
	for _, block := range result.Content {
		assert.Nil(t, block.CachePoint, "Converse ignores a cachePoint nested inside toolResult.content")
	}
	return result
}

func TestToolResultCacheControlBecomesSiblingCachePoint(t *testing.T) {
	req := convertChatHistory(t, agentLoopHistory(cachedToolText("main.go\nREADME.md")))

	assert.JSONEq(t, `{"role":"user","content":[
		{"toolResult":{"toolUseId":"call_1","content":[{"text":"main.go\nREADME.md"}],"status":"success"}},
		{"cachePoint":{"type":"default"}}
	]}`, lastConverseMessageJSON(t, req))
}

func TestToolResultCacheControlKeepsTTL(t *testing.T) {
	content := cachedToolText("main.go")
	content.ContentBlocks[0].CacheControl.TTL = new("1h")
	req := convertChatHistory(t, agentLoopHistory(content))

	last := req.Messages[len(req.Messages)-1]
	toolResultOf(t, last)
	require.Len(t, last.Content, 2)
	require.NotNil(t, last.Content[1].CachePoint)
	require.NotNil(t, last.Content[1].CachePoint.TTL)
	assert.Equal(t, "1h", string(*last.Content[1].CachePoint.TTL))
}

// Several marked blocks in one tool result still close one prefix: a single sibling checkpoint.
func TestToolResultWithSeveralMarkedBlocksGetsOneCachePoint(t *testing.T) {
	marker := &schemas.CacheControl{Type: schemas.CacheControlTypeEphemeral}
	req := convertChatHistory(t, agentLoopHistory(schemas.ChatMessageContent{ContentBlocks: []schemas.ChatContentBlock{
		{Type: schemas.ChatContentBlockTypeText, Text: new("part one"), CacheControl: marker},
		{Type: schemas.ChatContentBlockTypeText, Text: new("part two"), CacheControl: marker},
	}}))

	last := req.Messages[len(req.Messages)-1]
	assert.Len(t, toolResultOf(t, last).Content, 2)
	require.Len(t, last.Content, 2)
	assert.NotNil(t, last.Content[1].CachePoint)
}

// Bifrost's own prompt-cache injector promotes the newest message's string content to a single
// marked text block. Through the converter, that must land as a sibling checkpoint too.
func TestInjectedToolResultBreakpointReachesConverseAsSibling(t *testing.T) {
	plain := func() []schemas.ChatMessage {
		return agentLoopHistory(schemas.ChatMessageContent{ContentStr: new("main.go\nREADME.md")})
	}

	t.Run("last message", func(t *testing.T) {
		injected := providerUtils.InjectChatCacheBreakpoints(&schemas.PromptCacheConfig{
			AutoInject:      true,
			InjectionPoints: []schemas.CacheControlInjectionPoint{{Location: "message", Index: new(-1)}},
		}, plain())
		req := convertChatHistory(t, injected)

		assert.JSONEq(t, `{"role":"user","content":[
			{"toolResult":{"toolUseId":"call_1","content":[{"text":"main.go\nREADME.md"}],"status":"success"}},
			{"cachePoint":{"type":"default"}}
		]}`, lastConverseMessageJSON(t, req))
	})

	t.Run("system and last message", func(t *testing.T) {
		injected := providerUtils.InjectChatCacheBreakpoints(&schemas.PromptCacheConfig{
			AutoInject: true,
			InjectionPoints: []schemas.CacheControlInjectionPoint{
				{Location: "message", Role: new("system")},
				{Location: "message", Index: new(-1)},
			},
		}, plain())
		req := convertChatHistory(t, injected)

		require.Len(t, req.System, 2)
		assert.NotNil(t, req.System[1].CachePoint, "system breakpoint lands in Converse's system array")

		last := req.Messages[len(req.Messages)-1]
		toolResultOf(t, last)
		require.Len(t, last.Content, 2)
		assert.NotNil(t, last.Content[1].CachePoint)
	})
}

// When the breakpoint moves to a newer message on the next turn, the previously marked tool
// result becomes a plain string again. Its Converse representation must not change with the
// marker, or the cached prefix stops matching and every turn writes instead of reads.
func TestToolResultRepresentationDoesNotDependOnCacheMarker(t *testing.T) {
	for name, text := range map[string]string{
		"json object": `{"files": ["main.go", "README.md"]}`,
		"json array":  `["main.go", "README.md"]`,
		"plain text":  "main.go\nREADME.md",
	} {
		t.Run(name, func(t *testing.T) {
			asString := toolResultOf(t, convertChatHistory(t, agentLoopHistory(schemas.ChatMessageContent{ContentStr: new(text)})).Messages[2])
			marked := convertChatHistory(t, agentLoopHistory(cachedToolText(text)))
			asMarkedBlock := toolResultOf(t, marked.Messages[2])

			want, err := json.Marshal(asString.Content)
			require.NoError(t, err)
			got, err := json.Marshal(asMarkedBlock.Content)
			require.NoError(t, err)
			assert.JSONEq(t, string(want), string(got))
		})
	}
}

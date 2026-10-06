package bedrockmantle

import "testing"

func TestGPT6UsesOpenAIEndpoint(t *testing.T) {
	for _, model := range []string{"openai.gpt-6.1-sol", "openai.gpt-6-luna"} {
		for _, path := range []string{"responses", "chat/completions"} {
			t.Run(model+"/"+path, func(t *testing.T) {
				want := "https://bedrock-mantle.us-east-1.api.aws/openai/v1/" + path
				if got := mantleOpenAIURL(nil, "us-east-1", model, path); got != want {
					t.Fatalf("GPT-6 endpoint = %q, want %q", got, want)
				}
			})
		}
	}
}

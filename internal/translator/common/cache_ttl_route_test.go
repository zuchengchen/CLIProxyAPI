package common

import (
	"fmt"
	"strings"
	"testing"
)

func TestSplitDevinCacheCreationTTL_NoBreakpoints(t *testing.T) {
	body := []byte(`{"model":"claude-sonnet-5-5","messages":[{"role":"user","content":"hi"}]}`)
	five, one := SplitDevinCacheCreationTTL(body, 1000, false, "claude-sonnet-5-5")
	if five != 1000 || one != 0 {
		t.Fatalf("no breakpoint => all 5m, got 5m=%d 1h=%d", five, one)
	}
}

func TestSplitDevinCacheCreationTTL_UnspecifiedDefaultsTo5m(t *testing.T) {
	body := []byte(`{
		"system":[{"type":"text","text":"` + strings.Repeat("s", 4000) + `","cache_control":{"type":"ephemeral"}}],
		"messages":[{"role":"user","content":[{"type":"text","text":"hi","cache_control":{"type":"ephemeral"}}]}]
	}`)
	five, one := SplitDevinCacheCreationTTL(body, 800, true, "claude-sonnet-5-5")
	if five != 800 || one != 0 {
		t.Fatalf("unspecified ttl must default to 5m, got 5m=%d 1h=%d", five, one)
	}
}

func TestSplitDevinCacheCreationTTL_Explicit5mAnd1h(t *testing.T) {
	// system at 1h, message at 5m — valid order, split proportional to prefix.
	// Both blocks must clear the sonnet-5 1024-token minimum cacheable length.
	body := []byte(`{
		"system":[{"type":"text","text":"` + strings.Repeat("s", 6000) + `","cache_control":{"type":"ephemeral","ttl":"1h"}}],
		"messages":[{"role":"user","content":[{"type":"text","text":"` + strings.Repeat("m", 6000) + `","cache_control":{"type":"ephemeral","ttl":"5m"}}]}]
	}`)
	five, one := SplitDevinCacheCreationTTL(body, 800, true, "claude-sonnet-5-5")
	if five+one != 800 {
		t.Fatalf("split must conserve tokens: %d+%d != 800", five, one)
	}
	if one == 0 {
		t.Fatalf("explicit 1h breakpoint must produce a 1h share, got %d", one)
	}
	// system ~1000 tokens of ~2000 total => ~50% under 1h
	if one < 300 || one > 500 {
		t.Fatalf("1h share should be roughly half of 800, got %d", one)
	}
}

func TestSplitDevinCacheCreationTTL_InvalidOrderFallsBackTo5m(t *testing.T) {
	// 5m breakpoint before 1h — Anthropic would 400; metering reports all 5m.
	body := []byte(`{
		"system":[{"type":"text","text":"sys","cache_control":{"type":"ephemeral","ttl":"5m"}}],
		"messages":[{"role":"user","content":[{"type":"text","text":"hi","cache_control":{"type":"ephemeral","ttl":"1h"}}]}]
	}`)
	five, one := SplitDevinCacheCreationTTL(body, 500, true, "claude-sonnet-5-5")
	if five != 500 || one != 0 {
		t.Fatalf("invalid TTL order must fall back to 5m, got 5m=%d 1h=%d", five, one)
	}
}

func TestSplitDevinCacheCreationTTL_LongContextPromotion(t *testing.T) {
	// Prefix over 500k est tokens on an unspecified-ttl breakpoint promotes to 1h.
	big := strings.Repeat("x", 2_200_000) // ~550k est tokens
	body := []byte(fmt.Sprintf(`{
		"system":[{"type":"text","text":"%s","cache_control":{"type":"ephemeral"}}],
		"messages":[{"role":"user","content":[{"type":"text","text":"q","cache_control":{"type":"ephemeral"}}]}]
	}`, big))
	five, one := SplitDevinCacheCreationTTL(body, 10000, true, "claude-sonnet-5-5")
	if one == 0 {
		t.Fatalf("long-context prefix should promote to 1h, got 1h=%d", one)
	}
	// nearly the whole prefix is the big system block
	if one < 9000 {
		t.Fatalf("expected most of creation under 1h, got 1h=%d of %d", one, five+one)
	}
}

func TestSplitDevinCacheCreationTTL_SessionSuppressesSharedStaticRule(t *testing.T) {
	// A ~12k-token system prefix with no ttl: qualifies for R2 only when the
	// request has no session.
	big := strings.Repeat("x", 40_000) // ~10k est tokens
	body := []byte(fmt.Sprintf(`{
		"system":[{"type":"text","text":"%s","cache_control":{"type":"ephemeral"}}],
		"messages":[{"role":"user","content":[{"type":"text","text":"hi","cache_control":{"type":"ephemeral"}}]}]
	}`, big))

	fiveS, oneS := SplitDevinCacheCreationTTL(body, 1000, true, "claude-sonnet-5-5")
	if oneS != 0 {
		t.Fatalf("session-scoped request must not promote shared-static rule, got 1h=%d", oneS)
	}

	_, oneNo := SplitDevinCacheCreationTTL(body, 1000, false, "claude-sonnet-5-5")
	if oneNo == 0 {
		t.Fatalf("sessionless shared static prefix should promote to 1h, got 0")
	}
	_ = fiveS
}

func TestSplitDevinCacheCreationTTL_ZeroAndNegative(t *testing.T) {
	body := []byte(`{"messages":[{"role":"user","content":[{"type":"text","text":"hi","cache_control":{"type":"ephemeral","ttl":"1h"}}]}]}`)
	five, one := SplitDevinCacheCreationTTL(body, 0, true, "claude-sonnet-5-5")
	if five != 0 || one != 0 {
		t.Fatalf("zero creation must stay zero, got %d/%d", five, one)
	}
	five, one = SplitDevinCacheCreationTTL(body, -5, true, "claude-sonnet-5-5")
	if five != 0 || one != 0 {
		t.Fatalf("negative creation must clamp to zero, got %d/%d", five, one)
	}
}

func TestAnthropicViewUsage_ToolResultTailStaysCached(t *testing.T) {
	// An agentic-loop turn whose final message is a tool_result continuation
	// must not count it as fresh input — the whole prompt is rolling prefix.
	body := []byte(`{
		"system": [{"type":"text","text":"` + strings.Repeat("s", 4000) + `"}],
		"messages": [
			{"role":"user","content":"start"},
			{"role":"assistant","content":[{"type":"text","text":"ok"},{"type":"tool_use","id":"t1","name":"ls","input":{}}]},
			{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"files"}]}
		]
	}`)
	in, read, creation := AnthropicViewUsage(body, 4, 100000, 2000)
	if in != 4 {
		t.Fatalf("tool_result tail must not inflate input, got %d", in)
	}
	if read+creation+in != 4+100000+2000 {
		t.Fatalf("conservation broken: %d %d %d", in, read, creation)
	}
}

func TestAnthropicViewUsage_FreshUserTailMovesToInput(t *testing.T) {
	body := []byte(`{
		"system": [{"type":"text","text":"` + strings.Repeat("s", 4000) + `"}],
		"messages": [
			{"role":"user","content":"` + strings.Repeat("q", 400) + `"},
			{"role":"assistant","content":"ack"},
			{"role":"user","content":"` + strings.Repeat("x", 400) + `"}
		]
	}`)
	in, read, creation := AnthropicViewUsage(body, 0, 100000, 3000)
	if in <= 0 {
		t.Fatalf("fresh user tail must land in input, got %d", in)
	}
	if read+creation+in != 103000 {
		t.Fatalf("conservation broken: %d %d %d", in, read, creation)
	}
}

func TestSplitDevinCacheCreationTTL_MinCacheableSkipsShortBreakpoint(t *testing.T) {
	// sonnet-5 threshold is 1024 tokens; a breakpoint on a ~500-token block
	// must be silently skipped, so the write lands entirely in 5m.
	body := []byte(`{
		"system":[{"type":"text","text":"` + strings.Repeat("s", 1500) + `","cache_control":{"type":"ephemeral","ttl":"1h"}}],
		"messages":[{"role":"user","content":[{"type":"text","text":"` + strings.Repeat("m", 1500) + `","cache_control":{"type":"ephemeral"}}]}]
	}`)
	five, one := SplitDevinCacheCreationTTL(body, 400, true, "claude-sonnet-5-5")
	if one != 0 {
		t.Fatalf("sub-threshold 1h breakpoint must be skipped, got 1h=%d", one)
	}
	if five != 400 {
		t.Fatalf("creation must fall back to 5m, got 5m=%d", five)
	}
}

func TestSplitDevinCacheCreationTTL_TooManyBreakpoints(t *testing.T) {
	parts := make([]string, 5)
	for i := range parts {
		parts[i] = `{"role":"user","content":[{"type":"text","text":"` + strings.Repeat("x", 5000) + `","cache_control":{"type":"ephemeral","ttl":"1h"}}]}`
	}
	body := []byte(`{"messages":[` + strings.Join(parts, ",") + `]}`)
	five, one := SplitDevinCacheCreationTTL(body, 1000, true, "claude-sonnet-5-5")
	if five != 1000 || one != 0 {
		t.Fatalf(">4 breakpoints must refuse simulation -> all 5m, got 5m=%d 1h=%d", five, one)
	}
}

func TestSplitDevinCacheCreationTTL_InvalidCacheControlRefuses(t *testing.T) {
	body := []byte(`{
		"messages":[{"role":"user","content":[{"type":"text","text":"` + strings.Repeat("x", 8000) + `","cache_control":"not-an-object"}]}]
	}`)
	five, one := SplitDevinCacheCreationTTL(body, 600, true, "claude-sonnet-5-5")
	if five != 600 || one != 0 {
		t.Fatalf("invalid cache_control must refuse simulation, got 5m=%d 1h=%d", five, one)
	}
}

func TestSplitDevinCacheCreationTTL_BreakpointOnThinkingBlockRefuses(t *testing.T) {
	body := []byte(`{
		"messages":[{"role":"assistant","content":[
			{"type":"thinking","thinking":"` + strings.Repeat("t", 5000) + `","cache_control":{"type":"ephemeral","ttl":"1h"}},
			{"type":"text","text":"ok"}
		]},{"role":"user","content":"go"}]
	}`)
	five, one := SplitDevinCacheCreationTTL(body, 600, true, "claude-sonnet-5-5")
	if five != 600 || one != 0 {
		t.Fatalf("breakpoint on non-cacheable thinking block must refuse, got 5m=%d 1h=%d", five, one)
	}
}

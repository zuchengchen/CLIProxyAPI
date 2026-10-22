package common

import (
	"strings"

	"github.com/router-for-me/CLIProxyAPI/v8/internal/tokenestimate"
	"github.com/tidwall/gjson"
)

// Cache TTL metering helpers for upstreams that do not natively split cache
// writes by TTL (e.g. Devin, whose protobuf wire has no cache-control concept).
//
// Semantics follow Anthropic's prompt-caching model and the kiro.rs metering
// simulation: breakpoints are evaluated in tools -> system -> messages order,
// each declared cache_control.ttl is honored, and an unspecified ttl defaults
// to 5m unless a smart-routing rule promotes it to 1h. The result is a way to
// split an aggregate cache_write_tokens value into ephemeral_5m/1h buckets so
// relays that bill the two TTLs separately (e.g. sub2api) can keep doing so.

const (
	cacheTTL5m = "5m"
	cacheTTL1h = "1h"

	// maxCacheBreakpoints mirrors kiro.rs MAX_BREAKPOINTS: Anthropic rejects
	// requests carrying more than 4 breakpoints, so anything beyond that count
	// must not be simulated.
	maxCacheBreakpoints = 4

	// longContext1hMinTokens mirrors kiro.rs R1: a breakpoint whose cumulative
	// prefix already covers at least this many estimated tokens is worth the
	// 1h write premium, because re-creating it after a 5m expiry costs more
	// than the 0.75x upgrade while gaps between turns grow with context size.
	longContext1hMinTokens = 500_000

	// sharedStaticPrefix1hMinTokens mirrors kiro.rs R2: a breakpoint inside the
	// tools/system static prefix on a session-less request can be reused across
	// conversations, so a large enough prefix pays back the 1h write quickly.
	sharedStaticPrefix1hMinTokens = 8_192
)

// cacheBreakpoint records one evaluated cache_control breakpoint.
type cacheBreakpoint struct {
	ttl         string
	explicitTTL bool
	// cumulative estimated tokens of the prefix ending at this breakpoint,
	// in tools -> system -> messages evaluation order.
	cumTokens int64
	// static is true when the breakpoint sits inside tools or system.
	static bool
}

// estimateTokens approximates token count from plain text bytes; ~4 bytes per
// token matches the heuristic used elsewhere in this codebase.
func estimateTokens(raw string) int64 {
	return int64(len(raw) / tokenestimate.BytesPerToken)
}

// estimateBlockTokens estimates tokens for serialized JSON (a tool, system
// block, message content part, or whole request) in the same units as
// estimateTokens. Inline base64 images are charged by pixel area and other
// inline base64 blobs (e.g. PDFs) by a small fixed amount, so screenshots do
// not inflate the prefix into the long-context 1h routing tier.
func estimateBlockTokens(raw string) int64 {
	return tokenestimate.EstimateJSONTokensWithBlobs(raw)
}

// minCacheableTokens returns the smallest prefix length a breakpoint may
// cache for the given model, matching kiro.rs's per-family thresholds.
// Non-Claude models have no gate (returns 0).
func minCacheableTokens(model string) int64 {
	m := strings.ToLower(strings.TrimSpace(model))
	switch {
	case strings.Contains(m, "fable"), strings.Contains(m, "mythos"):
		return 512
	case strings.HasPrefix(m, "claude-haiku-4.5"), strings.HasPrefix(m, "claude-haiku-4-5"),
		strings.HasPrefix(m, "claude-opus-4.5"), strings.HasPrefix(m, "claude-opus-4-5"),
		strings.HasPrefix(m, "claude-opus-4.6"), strings.HasPrefix(m, "claude-opus-4-6"):
		return 4096
	case strings.HasPrefix(m, "claude-opus-4.7"), strings.HasPrefix(m, "claude-opus-4-7"),
		strings.HasPrefix(m, "claude-haiku-3"):
		return 2048
	case strings.HasPrefix(m, "claude-opus-5"):
		return 512
	case strings.HasPrefix(m, "claude-"):
		return 1024
	default:
		return 0
	}
}

// blockIsCacheable mirrors kiro.rs block_is_cacheable: thinking blocks and
// empty text blocks cannot host a breakpoint; every other typed block can.
// String content is normalized by the caller to a single text block first.
func blockIsCacheable(block gjson.Result) bool {
	if !block.IsObject() {
		return false
	}
	switch block.Get("type").String() {
	case "thinking", "redacted_thinking":
		return false
	case "text":
		return strings.TrimSpace(block.Get("text").String()) != ""
	case "":
		return false
	default:
		return true
	}
}

// cacheControlForBlock extracts a valid ephemeral cache_control from a block.
// Returns (ttl, explicitTTL, ok). A present but malformed cache_control
// (unparseable or non-ephemeral) reports ok=false so callers can refuse the
// whole request, matching kiro.rs's invalid_cache_control behavior.
func cacheControlForBlock(cc gjson.Result) (ttl string, explicit bool, ok bool) {
	if !cc.Exists() || cc.Type == gjson.Null {
		return "", false, false
	}
	if !cc.IsObject() || cc.Get("type").String() != "ephemeral" {
		return "", false, false
	}
	ttl = cc.Get("ttl").String()
	if ttl == "" {
		return cacheTTL5m, false, true
	}
	if !strings.EqualFold(ttl, cacheTTL5m) && !strings.EqualFold(ttl, cacheTTL1h) {
		return "", false, false
	}
	return strings.ToLower(ttl), true, true
}

// collectCacheBreakpoints walks a Claude-shaped request body and returns the
// declared cache_control breakpoints in evaluation order together with the
// cumulative estimated token counts at each breakpoint. Blocks that are not
// objects (string system prompts, string message content) still contribute
// their tokens to the running prefix.
//
// Returns (breakpoints, ok). ok is false when the request carries an invalid
// cache_control or a breakpoint on a non-cacheable block — the same
// conditions kiro.rs refuses to simulate.
func collectCacheBreakpoints(body []byte, model string) ([]cacheBreakpoint, bool) {
	minTokens := minCacheableTokens(model)
	var bps []cacheBreakpoint
	var cum int64
	valid := true

	markBreakpoint := func(cc gjson.Result, cacheable bool, static bool) {
		if !cc.Exists() || cc.Type == gjson.Null {
			return
		}
		ttl, explicit, ok := cacheControlForBlock(cc)
		if !ok || !cacheable {
			valid = false
			return
		}
		// Below the model's minimum cacheable length the breakpoint is silently
		// skipped — same as Anthropic and kiro.rs.
		if cum < minTokens {
			return
		}
		bps = append(bps, cacheBreakpoint{ttl: ttl, explicitTTL: explicit, cumTokens: cum, static: static})
	}

	// tools
	if tools := gjson.GetBytes(body, "tools"); tools.IsArray() {
		tools.ForEach(func(_, tool gjson.Result) bool {
			cum += estimateBlockTokens(tool.Raw)
			markBreakpoint(tool.Get("cache_control"), true, true)
			return true
		})
	}

	// system
	system := gjson.GetBytes(body, "system")
	switch {
	case system.IsArray():
		system.ForEach(func(_, block gjson.Result) bool {
			cum += estimateBlockTokens(block.Raw)
			markBreakpoint(block.Get("cache_control"), blockIsCacheable(block), true)
			return true
		})
	case system.Type == gjson.String:
		cum += estimateTokens(system.String())
		// A cache_control cannot attach to a bare string system prompt; if the
		// client sent one alongside, it is invalid input.
		if cc := gjson.GetBytes(body, "system_cache_control"); cc.Exists() {
			valid = false
		}
	}

	// messages
	if messages := gjson.GetBytes(body, "messages"); messages.IsArray() {
		messages.ForEach(func(_, msg gjson.Result) bool {
			content := msg.Get("content")
			if content.IsArray() {
				content.ForEach(func(_, part gjson.Result) bool {
					cum += estimateBlockTokens(part.Raw)
					markBreakpoint(part.Get("cache_control"), blockIsCacheable(part), false)
					return true
				})
			} else {
				cum += estimateBlockTokens(content.Raw)
			}
			// A message-level cache_control applies to the whole message's
			// prefix; treat it as a breakpoint at this position.
			if msgCC := msg.Get("cache_control"); msgCC.Exists() {
				markBreakpoint(msgCC, true, false)
			}
			return true
		})
	}

	if !valid || len(bps) > maxCacheBreakpoints {
		return nil, false
	}
	return bps, true
}

// applySmartTTLRouting promotes unspecified-ttl breakpoints to 1h following
// the kiro.rs rules, keeping all 1h breakpoints ahead of any 5m breakpoint:
// find the deepest promotable breakpoint, promote it and every unspecified-ttl
// breakpoint before it, and stop at the first explicit-5m breakpoint.
func applySmartTTLRouting(bps []cacheBreakpoint, hasSession bool) []cacheBreakpoint {
	target := -1
	for i, bp := range bps {
		if bp.explicitTTL && bp.ttl == cacheTTL5m {
			// An explicit 5m breakpoint ends the promotion run.
			break
		}
		if bp.explicitTTL {
			continue
		}
		longContext := bp.cumTokens >= longContext1hMinTokens
		sharedStatic := !hasSession && bp.static && bp.cumTokens >= sharedStaticPrefix1hMinTokens
		if longContext || sharedStatic {
			target = i
		}
	}
	if target < 0 {
		return bps
	}
	out := make([]cacheBreakpoint, len(bps))
	copy(out, bps)
	for i := 0; i <= target; i++ {
		if !out[i].explicitTTL {
			out[i].ttl = cacheTTL1h
		}
	}
	return out
}

// normalizeTTLOrderForMetering reports whether declared breakpoints satisfy
// Anthropic's "all 1h breakpoints precede any 5m breakpoint" ordering. When the
// client violates it, metering falls back to treating everything as 5m — the
// same conservative outcome Anthropic's 400 produces.
func ttlOrderValid(bps []cacheBreakpoint) bool {
	seen5m := false
	for _, bp := range bps {
		if bp.ttl == cacheTTL5m {
			seen5m = true
			continue
		}
		if seen5m {
			return false
		}
	}
	return true
}

// cache1hFraction computes the fraction of the deepest cached prefix that
// sits under a 1h breakpoint. The prefix ends at the last breakpoint; all 1h
// breakpoints form a contiguous run from the start after routing, so the 1h
// share is the deepest 1h breakpoint's cumulative position over the deepest
// breakpoint's cumulative position.
func cache1hFraction(bps []cacheBreakpoint) float64 {
	if len(bps) == 0 {
		return 0
	}
	total := bps[len(bps)-1].cumTokens
	if total <= 0 {
		return 0
	}
	var oneHour int64
	for _, bp := range bps {
		if bp.ttl == cacheTTL1h && bp.cumTokens > oneHour {
			oneHour = bp.cumTokens
		}
	}
	frac := float64(oneHour) / float64(total)
	if frac < 0 {
		return 0
	}
	if frac > 1 {
		return 1
	}
	return frac
}

// SplitDevinCacheCreationTTL splits an aggregate cache_creation token count
// reported by a TTL-unaware upstream into (ephemeral_5m, ephemeral_1h) buckets,
// based on the breakpoints declared in the original Claude request body and
// the smart-routing rules above. hasSession should be true whenever the
// request carried any session identifier (metadata.user_id session, explicit
// session fields, or an affinity header) because session-scoped keys are never
// shared across conversations and therefore never qualify for rule R2.
//
// When the client declared a mixed order that violates the 1h-before-5m
// constraint, carries an invalid cache_control, or declares more than four
// breakpoints, the whole write is reported as 5m — the conservative outcome
// Anthropic's rejection produces.
func SplitDevinCacheCreationTTL(originalRequest []byte, cacheCreationTokens int64, hasSession bool, model string) (int64, int64) {
	creation := cacheCreationTokens
	if creation < 0 {
		creation = 0
	}
	if creation == 0 || len(originalRequest) == 0 {
		return creation, 0
	}
	bps, ok := collectCacheBreakpoints(originalRequest, model)
	if !ok || len(bps) == 0 {
		return creation, 0
	}
	bps = applySmartTTLRouting(bps, hasSession)
	if !ttlOrderValid(bps) {
		return creation, 0
	}
	frac := cache1hFraction(bps)
	oneHour := int64(float64(creation)*frac + 0.5)
	if oneHour < 0 {
		oneHour = 0
	}
	if oneHour > creation {
		oneHour = creation
	}
	return creation - oneHour, oneHour
}

// currentTurnTailTokens estimates the tokens of the trailing current-turn
// input — the part Anthropic never counts as cached. Following kiro.rs's
// automatic-breakpoint rule, the rolling breakpoint lands on the last
// cacheable block that is NOT the current-turn input:
//
//	current-turn input = the final message, only when it is a user turn
//	that does NOT carry tool_result blocks.
//
// A tool_result-bearing final user message is a continuation inside an
// agentic loop, not a fresh human turn, so it stays inside the cached prefix
// and the tail is empty. When the last message is not a user turn at all the
// tail is likewise empty — the whole prompt belongs to the rolling prefix.
func currentTurnTailTokens(body []byte) int64 {
	messages := gjson.GetBytes(body, "messages")
	if !messages.IsArray() {
		return 0
	}
	arr := messages.Array()
	if len(arr) == 0 {
		return 0
	}
	last := arr[len(arr)-1]
	if last.Get("role").String() != "user" {
		return 0
	}
	var tail int64
	var hasToolResult bool
	content := last.Get("content")
	if content.IsArray() {
		content.ForEach(func(_, part gjson.Result) bool {
			if part.Get("type").String() == "tool_result" {
				hasToolResult = true
			}
			tail += estimateBlockTokens(part.Raw)
			return true
		})
	} else {
		tail += estimateBlockTokens(content.Raw)
	}
	if hasToolResult {
		return 0
	}
	return tail
}

// AnthropicViewUsage remaps an upstream's aggregate cache accounting into the
// Anthropic billing view where the trailing current-turn input is reported as
// plain input_tokens instead of being absorbed into cache writes. Some
// upstreams (e.g. Devin) charge the whole prompt including the fresh tail as
// cache writes; Anthropic's model only writes the stable rolling prefix, so
// relays that bill Anthropic-style should see the tail moved back to input.
//
// Inputs are the upstream-reported totals: uncached (prompt_tokens), cached
// (cache read), cacheWrite (cache creation), and the original request body
// used to estimate the current-turn tail share.
//
// Returns (inputTokens, cacheRead, cacheCreation); output tokens pass through
// unchanged and are handled by the caller. Conservation holds:
// input + read + creation == uncached + cached + cacheWrite.
func AnthropicViewUsage(originalRequest []byte, uncached, cached, cacheWrite int64) (input, read, creation int64) {
	total := uncached + cached + cacheWrite
	if total <= 0 {
		return 0, 0, 0
	}

	// Estimated shares of the current-turn tail within the whole prompt.
	// tailReal scales the estimated tail into the real upstream total so the
	// split is proportional rather than dependent on the estimator's absolute
	// accuracy.
	var tailReal int64
	if len(originalRequest) > 0 {
		tailEst := currentTurnTailTokens(originalRequest)
		if tailEst > 0 {
			promptEst := estimateBlockTokens(string(originalRequest))
			if promptEst > tailEst {
				tailReal = total * tailEst / promptEst
			}
		}
	}
	if tailReal < 0 {
		tailReal = 0
	}
	if tailReal > total {
		tailReal = total
	}

	// Reallocate only the current-turn share inside the cache write back to
	// input. Upstream's read and write stay authoritative; nothing is
	// fabricated — if the upstream wrote nothing, creation stays zero even
	// when the read prefix was large.
	tailFromWrite := tailReal
	if tailFromWrite > cacheWrite {
		tailFromWrite = cacheWrite
	}

	input = uncached + tailFromWrite
	read = cached
	if read < 0 {
		read = 0
	}
	creation = cacheWrite - tailFromWrite
	if creation < 0 {
		creation = 0
	}
	return input, read, creation
}

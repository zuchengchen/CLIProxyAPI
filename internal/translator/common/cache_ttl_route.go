package common

import (
	"strings"

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

// estimateTokens approximates token count from serialized bytes; ~4 bytes per
// token matches the heuristic used elsewhere in this codebase.
func estimateTokens(raw string) int64 {
	return int64(len(raw) / 4)
}

// collectCacheBreakpoints walks a Claude-shaped request body and returns the
// declared cache_control breakpoints in evaluation order together with the
// cumulative estimated token counts at each breakpoint. Blocks that are not
// objects (string system prompts, string message content) still contribute
// their tokens to the running prefix.
func collectCacheBreakpoints(body []byte) []cacheBreakpoint {
	var bps []cacheBreakpoint
	var cum int64

	markBreakpoint := func(cc gjson.Result, static bool) {
		if !cc.Exists() || cc.Get("type").String() != "ephemeral" {
			return
		}
		ttl := cc.Get("ttl").String()
		explicit := ttl != ""
		if !explicit {
			ttl = cacheTTL5m
		} else if !strings.EqualFold(ttl, cacheTTL5m) && !strings.EqualFold(ttl, cacheTTL1h) {
			// Unknown ttl is treated as 5m for metering rather than dropped;
			// the upstream ignored it anyway.
			ttl = cacheTTL5m
		}
		bps = append(bps, cacheBreakpoint{ttl: ttl, explicitTTL: explicit, cumTokens: cum, static: static})
	}

	// tools
	if tools := gjson.GetBytes(body, "tools"); tools.IsArray() {
		tools.ForEach(func(_, tool gjson.Result) bool {
			cum += estimateTokens(tool.Raw)
			markBreakpoint(tool.Get("cache_control"), true)
			return true
		})
	}

	// system
	system := gjson.GetBytes(body, "system")
	switch {
	case system.IsArray():
		system.ForEach(func(_, block gjson.Result) bool {
			cum += estimateTokens(block.Raw)
			markBreakpoint(block.Get("cache_control"), true)
			return true
		})
	case system.Type == gjson.String:
		cum += estimateTokens(system.String())
	}

	// messages
	if messages := gjson.GetBytes(body, "messages"); messages.IsArray() {
		messages.ForEach(func(_, msg gjson.Result) bool {
			content := msg.Get("content")
			if content.IsArray() {
				content.ForEach(func(_, part gjson.Result) bool {
					cum += estimateTokens(part.Raw)
					markBreakpoint(part.Get("cache_control"), false)
					return true
				})
			} else {
				cum += estimateTokens(content.Raw)
			}
			// A message-level cache_control applies to the whole message's
			// prefix; treat it as a breakpoint at this position.
			if msgCC := msg.Get("cache_control"); msgCC.Exists() {
				markBreakpoint(msgCC, false)
			}
			return true
		})
	}

	return bps
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
// constraint the whole write is reported as 5m, matching the conservative
// outcome of a rejected request.
func SplitDevinCacheCreationTTL(originalRequest []byte, cacheCreationTokens int64, hasSession bool) (int64, int64) {
	creation := cacheCreationTokens
	if creation < 0 {
		creation = 0
	}
	if creation == 0 || len(originalRequest) == 0 {
		return creation, 0
	}
	bps := collectCacheBreakpoints(originalRequest)
	if len(bps) == 0 {
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

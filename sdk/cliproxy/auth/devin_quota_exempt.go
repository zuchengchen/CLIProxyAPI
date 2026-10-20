package auth

import (
	"strings"
	"time"
)

// Devin SWE-2 is quota-free on self-serve plans, so a daily/weekly usage
// exhaustion reported by a paid sibling (Claude, etc.) must not cool it
// down. Paid models keep the credential-wide cooldown from issue #6690.

func isDevinProvider(provider string) bool {
	return strings.EqualFold(strings.TrimSpace(provider), "devin")
}

func bareDevinModelID(model string) string {
	key := strings.ToLower(canonicalModelKey(model))
	if i := strings.LastIndex(key, "/"); i >= 0 {
		key = key[i+1:]
	}
	return key
}

func isDevinQuotaExemptModel(provider, model string) bool {
	if model == "" || !isDevinProvider(provider) {
		return false
	}
	return bareDevinModelID(model) == "swe-2"
}

// isDevinCredentialQuotaPropagationOnly reports whether a model state is
// blocked only because a sibling's daily/weekly quota was copied onto it.
// The reporting model of that event uses reason "quota" and stays blocked.
func isDevinCredentialQuotaPropagationOnly(state *ModelState, now time.Time) bool {
	if state == nil {
		return false
	}
	reason := strings.TrimSpace(state.Quota.Reason)
	if reason != "credential_quota" && !state.credentialQuotaPropagated {
		return false
	}
	if reason == "quota" {
		return false
	}
	if shadow := state.credentialQuotaShadow; shadow != nil && isModelStateActiveCooldown(shadow, now) {
		return false
	}
	return true
}

package auth

import (
	"context"
	"net/http"
	"testing"
	"time"

	"github.com/router-for-me/CLIProxyAPI/v8/internal/registry"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v8/sdk/cliproxy/executor"
)

func TestIsDevinQuotaExemptModel(t *testing.T) {
	t.Parallel()
	cases := []struct {
		provider string
		model    string
		want     bool
	}{
		{"devin", "swe-2", true},
		{"devin", "devin/swe-2", true},
		{"Devin", "devin/swe-2(high)", true},
		{"devin", "claude-opus-5-5", false},
		{"devin", "devin/claude-sonnet-5-5", false},
		{"claude", "swe-2", false},
		{"devin", "", false},
	}
	for _, tc := range cases {
		if got := isDevinQuotaExemptModel(tc.provider, tc.model); got != tc.want {
			t.Errorf("isDevinQuotaExemptModel(%q, %q) = %v, want %v", tc.provider, tc.model, got, tc.want)
		}
	}
}

func devinSwe2AuthFixture(t *testing.T, id string) (*Manager, *Auth) {
	t.Helper()
	m := NewManager(nil, nil, nil)
	auth := &Auth{ID: id, Provider: "devin", Status: StatusActive}
	registry.GetGlobalRegistry().RegisterClient(auth.ID, "devin", []*registry.ModelInfo{
		{ID: "claude-opus-5-5"},
		{ID: "claude-sonnet-5-5"},
		{ID: "swe-2"},
		{ID: "devin/swe-2"},
	})
	t.Cleanup(func() { registry.GetGlobalRegistry().UnregisterClient(auth.ID) })
	if _, err := m.Register(context.Background(), auth); err != nil {
		t.Fatalf("register: %v", err)
	}
	return m, auth
}

func markDevinWeeklyQuota(t *testing.T, m *Manager, authID, model string) {
	t.Helper()
	retryAfter := 72 * time.Hour
	m.MarkResult(context.Background(), Result{
		AuthID:          authID,
		Provider:        "devin",
		Model:           model,
		Success:         false,
		RetryAfter:      &retryAfter,
		CredentialScope: true,
		Error:           &Error{Code: "rate_limit", Message: "Your weekly usage quota has been exhausted.", HTTPStatus: http.StatusTooManyRequests},
	})
}

func TestDevinWeeklyQuotaDoesNotBlockSwe2(t *testing.T) {
	withQuotaCooldownEnabled(t)
	m, auth := devinSwe2AuthFixture(t, "devin-swe2-exempt-weekly")

	// swe-2 already has a live model state, matching production accounts.
	m.MarkResult(context.Background(), Result{
		AuthID: auth.ID, Provider: "devin", Model: "devin/swe-2", Success: true,
	})
	markDevinWeeklyQuota(t, m, auth.ID, "claude-sonnet-5-5")

	stored, _ := m.GetByID(auth.ID)
	if stored == nil || stored.Quota.Reason != "credential_quota" {
		t.Fatalf("expected credential_quota on auth, got %+v", stored)
	}
	if blocked, _, _ := isAuthBlockedForModel(stored, "claude-sonnet-5-5", time.Now()); !blocked {
		t.Fatal("paid Claude model must stay blocked by weekly quota")
	}
	if blocked, _, _ := isAuthBlockedForModel(stored, "claude-opus-5-5", time.Now()); !blocked {
		t.Fatal("sibling paid model must stay blocked by credential quota")
	}
	for _, model := range []string{"swe-2", "devin/swe-2"} {
		if blocked, reason, _ := isAuthBlockedForModel(stored, model, time.Now()); blocked {
			t.Fatalf("%s must remain selectable after sibling weekly quota, reason=%v", model, reason)
		}
	}
	sweState := existingModelState(stored, canonicalModelKey("devin/swe-2"))
	if sweState != nil && (sweState.credentialQuotaPropagated || sweState.Quota.Reason == "credential_quota") {
		t.Fatalf("weekly quota must not wrap quota-free swe-2, got %+v", sweState)
	}
	picked, errPick := m.Selector().Pick(context.Background(), "devin", "devin/swe-2", cliproxyexecutor.Options{}, []*Auth{stored})
	if errPick != nil || picked == nil || picked.ID != auth.ID {
		t.Fatalf("selector must pick swe-2, err=%v picked=%v", errPick, picked)
	}
}

func TestDevinLegacySwe2WrapStillSelectable(t *testing.T) {
	withQuotaCooldownEnabled(t)
	m, auth := devinSwe2AuthFixture(t, "devin-swe2-legacy-wrap")
	recoverAt := time.Now().Add(18 * time.Hour)

	stored, _ := m.GetByID(auth.ID)
	stored.Unavailable = true
	stored.Status = StatusError
	stored.NextRetryAfter = recoverAt
	stored.Quota = QuotaState{Exceeded: true, Reason: "credential_quota", NextRecoverAt: recoverAt}
	stored.ModelStates = map[string]*ModelState{
		"devin/claude-sonnet-5-5": {
			Status: StatusError, Unavailable: true, NextRetryAfter: recoverAt,
			Quota: QuotaState{Exceeded: true, Reason: "quota", NextRecoverAt: recoverAt},
		},
		"devin/swe-2": {
			Status: StatusError, Unavailable: true, NextRetryAfter: recoverAt,
			Quota:                     QuotaState{Exceeded: true, Reason: "credential_quota", NextRecoverAt: recoverAt},
			credentialQuotaPropagated: true,
		},
	}
	if _, err := m.Update(context.Background(), stored); err != nil {
		t.Fatalf("seed wrapped auth: %v", err)
	}
	seeded, _ := m.GetByID(auth.ID)
	if blocked, _, _ := isAuthBlockedForModel(seeded, "devin/swe-2", time.Now()); blocked {
		t.Fatal("legacy credential_quota wrap on swe-2 must not block selection")
	}
	if blocked, _, _ := isAuthBlockedForModel(seeded, "swe-2", time.Now()); blocked {
		t.Fatal("bare swe-2 must not inherit a legacy wrap")
	}
	if blocked, _, _ := isAuthBlockedForModel(seeded, "devin/claude-sonnet-5-5", time.Now()); !blocked {
		t.Fatal("reporting Claude model must remain blocked")
	}
}

func TestDevinSwe2OwnQuotaStillBlocks(t *testing.T) {
	withQuotaCooldownEnabled(t)
	m, auth := devinSwe2AuthFixture(t, "devin-swe2-own-quota")
	retryAfter := 18 * time.Hour
	m.MarkResult(context.Background(), Result{
		AuthID:          auth.ID,
		Provider:        "devin",
		Model:           "devin/swe-2",
		Success:         false,
		RetryAfter:      &retryAfter,
		CredentialScope: true,
		Error:           &Error{Code: "rate_limit", Message: "Your daily usage quota has been exhausted.", HTTPStatus: http.StatusTooManyRequests},
	})
	stored, _ := m.GetByID(auth.ID)
	if blocked, _, _ := isAuthBlockedForModel(stored, "devin/swe-2", time.Now()); !blocked {
		t.Fatal("swe-2 must block when it is the reporting model of a quota exhaustion")
	}
}

func TestDevinSwe2Independent404SurvivesQuota(t *testing.T) {
	withQuotaCooldownEnabled(t)
	m, auth := devinSwe2AuthFixture(t, "devin-swe2-independent-404")
	m.MarkResult(context.Background(), Result{
		AuthID: auth.ID, Provider: "devin", Model: "devin/swe-2", Success: false,
		Error: &Error{Code: "not_found", Message: "model missing", HTTPStatus: http.StatusNotFound},
	})
	markDevinWeeklyQuota(t, m, auth.ID, "claude-opus-5-5")
	stored, _ := m.GetByID(auth.ID)
	if blocked, _, _ := isAuthBlockedForModel(stored, "devin/swe-2", time.Now()); !blocked {
		t.Fatal("independent swe-2 404 must remain blocking after sibling quota")
	}
	sweState := existingModelState(stored, canonicalModelKey("devin/swe-2"))
	if sweState == nil {
		t.Fatal("swe-2 state missing")
	}
	if sweState.credentialQuotaPropagated || sweState.Quota.Reason == "credential_quota" {
		t.Fatalf("quota-free swe-2 must keep its independent 404, not a credential wrap: %+v", sweState)
	}
}

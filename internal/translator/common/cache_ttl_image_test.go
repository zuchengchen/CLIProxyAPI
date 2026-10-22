package common

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"image"
	"image/png"
	"math/rand"
	"strings"
	"testing"
)

// noisyPNGBase64 builds an incompressible PNG so the base64 size resembles a
// real screenshot while the pixel area stays small.
func noisyPNGBase64(t *testing.T, w, h int) string {
	t.Helper()
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	rng := rand.New(rand.NewSource(1))
	rng.Read(img.Pix)
	for i := 3; i < len(img.Pix); i += 4 {
		img.Pix[i] = 255
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatalf("encode png: %v", err)
	}
	return base64.StdEncoding.EncodeToString(buf.Bytes())
}

func TestEstimateBlockTokens_ImageChargedByPixelArea(t *testing.T) {
	data := noisyPNGBase64(t, 700, 900)
	block := fmt.Sprintf(`{"type":"image","source":{"type":"base64","media_type":"image/png","data":%q}}`, data)
	got := estimateBlockTokens(block)
	want := int64(len(block)-len(data)-2)/4 + 700*900/750
	if got != want {
		t.Fatalf("image block estimate = %d, want %d (naive len/4 = %d)", got, want, len(block)/4)
	}
}

func TestEstimateBlockTokens_TextUnchanged(t *testing.T) {
	for _, raw := range []string{
		`{"type":"text","text":"` + strings.Repeat("hello ", 1000) + `"}`,
		`"plain string content"`,
		`{"name":"ls","description":"list","input_schema":{"type":"object"}}`,
	} {
		if got := estimateBlockTokens(raw); got != estimateTokens(raw) {
			t.Fatalf("text estimate for %.40q = %d, want %d", raw, got, estimateTokens(raw))
		}
	}
}

func TestEstimateBlockTokens_InlinePDFCapped(t *testing.T) {
	pdf := base64.StdEncoding.EncodeToString(bytes.Repeat([]byte("%PDF-1.7 stream"), 50_000))
	block := fmt.Sprintf(`{"type":"document","source":{"type":"base64","media_type":"application/pdf","data":%q}}`, pdf)
	if got := estimateBlockTokens(block); got > 1700 {
		t.Fatalf("pdf block estimate = %d, want <= 1700 (naive len/4 = %d)", got, len(block)/4)
	}
}

func TestSplitDevinCacheCreationTTL_LargeScreenshotDoesNotPromoteTo1h(t *testing.T) {
	// ~2.4 MB of base64 (~600k tokens by length) but only 1000x600 pixels.
	data := noisyPNGBase64(t, 1000, 600)
	if len(data)/4 < longContext1hMinTokens {
		t.Fatalf("fixture too small to trip the length heuristic: %d bytes", len(data))
	}
	request := func(toolContent string) []byte {
		return []byte(fmt.Sprintf(`{
		"system":[{"type":"text","text":"%s"}],
		"messages":[
			{"role":"user","content":"take a screenshot"},
			{"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"screenshot","input":{}}]},
			{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":[%s],"cache_control":{"type":"ephemeral"}}]}
		]
	}`, strings.Repeat("s", 8000), toolContent))
	}
	body := request(fmt.Sprintf(`{"type":"image","source":{"type":"base64","media_type":"image/png","data":%q}}`, data))

	five, one := SplitDevinCacheCreationTTL(body, 10000, true, "claude-sonnet-5-5")
	if five != 10000 || one != 0 {
		t.Fatalf("screenshot-only payload must stay 5m, got 5m=%d 1h=%d", five, one)
	}

	// Control: the same byte volume as text still promotes to 1h.
	textBody := request(fmt.Sprintf(`{"type":"text","text":%q}`, strings.Repeat("x", len(data))))
	if _, oneText := SplitDevinCacheCreationTTL(textBody, 10000, true, "claude-sonnet-5-5"); oneText == 0 {
		t.Fatalf("equal-size text payload should still promote to 1h")
	}
}

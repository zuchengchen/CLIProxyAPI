package tokenestimate

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math/rand"
	"strings"
	"testing"
)

// noisyPNGBase64 builds an incompressible PNG so the base64 size resembles a
// real screenshot (hundreds of KB) while the pixel area stays small.
func noisyPNGBase64(t *testing.T, w, h int) string {
	t.Helper()
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	rng := rand.New(rand.NewSource(1))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.Set(x, y, color.NRGBA{uint8(rng.Intn(256)), uint8(rng.Intn(256)), uint8(rng.Intn(256)), 255})
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatalf("encode png: %v", err)
	}
	return base64.StdEncoding.EncodeToString(buf.Bytes())
}

func TestEstimateImageTokens_UsesPixelArea(t *testing.T) {
	data := noisyPNGBase64(t, 700, 900)
	if len(data) < 1_000_000 {
		t.Fatalf("fixture too small: %d bytes", len(data))
	}
	got := EstimateImageTokens(data)
	if got != 700*900/750 {
		t.Fatalf("EstimateImageTokens = %d, want %d", got, 700*900/750)
	}
	if dataURL := EstimateImageTokens("data:image/png;base64," + data); dataURL != got {
		t.Fatalf("data URL estimate = %d, want %d", dataURL, got)
	}
}

func TestEstimateImageTokens_ResizesAndFallsBack(t *testing.T) {
	if got := imageTokensForDimensions(8000, 8000); got > imageTokenMaxPixels/imageTokenPixelDivisor+1 {
		t.Fatalf("huge image tokens = %d, want <= %d", got, imageTokenMaxPixels/imageTokenPixelDivisor)
	}
	if got := imageTokensForDimensions(10, 10); got != imageTokenMinimum {
		t.Fatalf("tiny image tokens = %d, want %d", got, imageTokenMinimum)
	}
	if got := EstimateImageTokens("UklGRiQAAABXRUJQVlA4IBgAAAAwAQCdASoBAAEAAwA0JaQAA3AA/vuUAAA="); got != imageTokenFallback {
		t.Fatalf("webp fallback = %d, want %d", got, imageTokenFallback)
	}
	if got := EstimateImageTokens(""); got != 0 {
		t.Fatalf("empty image = %d, want 0", got)
	}
}

func TestEstimateJSONPromptTokens_ExcludesInlineImageBytes(t *testing.T) {
	data := noisyPNGBase64(t, 700, 900)
	text := "hello world, describe this screenshot"
	claude := []byte(fmt.Sprintf(`{"messages":[{"role":"user","content":[{"type":"text","text":%q},{"type":"image","source":{"type":"base64","media_type":"image/png","data":%q}}]}]}`, text, data))
	got := EstimateJSONPromptTokens(claude)
	textOnly := int64(len(claude)-len(data)-2) / 4
	want := textOnly + 840
	if got != want {
		t.Fatalf("claude estimate = %d, want %d (naive len/4 = %d)", got, want, len(claude)/4)
	}

	openai := []byte(fmt.Sprintf(`{"input":[{"role":"user","content":[{"type":"input_image","image_url":"data:image/png;base64,%s"}]}]}`, data))
	if got := EstimateJSONPromptTokens(openai); got > 1000 {
		t.Fatalf("data URL estimate = %d, want < 1000", got)
	}

	plain := []byte(`{"messages":[{"role":"user","content":"data: not an image"}],"data":"x"}`)
	if got := EstimateJSONPromptTokens(plain); got != int64(len(plain)/4) {
		t.Fatalf("plain estimate = %d, want %d", got, len(plain)/4)
	}
}

func TestEstimateJSONTokensWithBlobs(t *testing.T) {
	data := noisyPNGBase64(t, 700, 900)
	pdf := base64.StdEncoding.EncodeToString(bytes.Repeat([]byte("%PDF-1.7 stream"), 50_000))

	image := fmt.Sprintf(`{"type":"image","source":{"type":"base64","media_type":"image/png","data":%q}}`, data)
	if got, want := EstimateJSONTokensWithBlobs(image), EstimateJSONPromptTokens([]byte(image)); got != want || got > 1000 {
		t.Fatalf("image estimate = %d, want %d (< 1000)", got, want)
	}

	doc := fmt.Sprintf(`{"type":"document","source":{"type":"base64","media_type":"application/pdf","data":%q}}`, pdf)
	if got, want := EstimateJSONTokensWithBlobs(doc), int64(len(doc)-len(pdf)-2)/4+InlineBlobTokens; got != want {
		t.Fatalf("pdf estimate = %d, want %d (naive len/4 = %d)", got, want, len(doc)/4)
	}
	if got := EstimateJSONPromptTokens([]byte(doc)); got != int64(len(doc)/4) {
		t.Fatalf("EstimateJSONPromptTokens must keep pdf on length heuristic, got %d", got)
	}

	fileURL := fmt.Sprintf(`{"type":"input_file","file_data":"data:application/pdf;base64,%s"}`, pdf)
	if got := EstimateJSONTokensWithBlobs(fileURL); got > InlineBlobTokens+100 {
		t.Fatalf("data URL blob estimate = %d, want <= %d", got, InlineBlobTokens+100)
	}

	gemini := fmt.Sprintf(`{"inlineData":{"mimeType":"application/pdf","data":%q}}`, pdf)
	if got := EstimateJSONTokensWithBlobs(gemini); got > InlineBlobTokens+100 {
		t.Fatalf("gemini inlineData estimate = %d, want <= %d", got, InlineBlobTokens+100)
	}

	tinyBlob := `{"type":"document","source":{"type":"base64","media_type":"application/pdf","data":"QUJD"}}`
	if got := EstimateJSONTokensWithBlobs(tinyBlob); got != int64(len(tinyBlob)/4) {
		t.Fatalf("tiny blob estimate = %d, want %d", got, len(tinyBlob)/4)
	}

	for _, raw := range []string{
		`{"type":"text","text":"data: not a blob, just text"}`,
		`{"type":"document","source":{"type":"text","media_type":"text/plain","data":"` + strings.Repeat("plain text ", 2000) + `"}}`,
		`"a bare string content"`,
	} {
		if got := EstimateJSONTokensWithBlobs(raw); got != int64(len(raw)/4) {
			t.Fatalf("text estimate for %.40q = %d, want %d", raw, got, len(raw)/4)
		}
	}
}

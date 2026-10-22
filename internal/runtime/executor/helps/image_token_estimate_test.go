package helps

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math/rand"
	"testing"

	"google.golang.org/protobuf/encoding/protowire"
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

func TestEstimateDevinWirePromptTokens_ExcludesImageBytes(t *testing.T) {
	data := noisyPNGBase64(t, 700, 900)
	var img []byte
	img = protowire.AppendTag(img, 1, protowire.BytesType)
	img = protowire.AppendString(img, data)
	img = protowire.AppendTag(img, 2, protowire.BytesType)
	img = protowire.AppendString(img, "image/png")
	var prompt []byte
	prompt = protowire.AppendTag(prompt, 3, protowire.BytesType)
	prompt = protowire.AppendString(prompt, "describe")
	prompt = protowire.AppendTag(prompt, 10, protowire.BytesType)
	prompt = protowire.AppendBytes(prompt, img)
	var wire []byte
	wire = protowire.AppendTag(wire, 2, protowire.BytesType)
	wire = protowire.AppendString(wire, "system")
	wire = protowire.AppendTag(wire, 3, protowire.BytesType)
	wire = protowire.AppendBytes(wire, prompt)

	got := EstimateDevinWirePromptTokens(wire)
	want := int64(len(wire)-len(data))/4 + 840
	if got != want {
		t.Fatalf("wire estimate = %d, want %d (naive len/4 = %d)", got, want, len(wire)/4)
	}
	if got := EstimateDevinWirePromptTokens([]byte{0xff, 0xff}); got != 0 {
		t.Fatalf("malformed wire estimate = %d, want 0", got)
	}
}

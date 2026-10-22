package helps

import (
	"bytes"
	"encoding/base64"
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

package helps

import (
	"encoding/base64"
	"image"
	_ "image/gif"  // register GIF header decoder for image.DecodeConfig
	_ "image/jpeg" // register JPEG header decoder for image.DecodeConfig
	_ "image/png"  // register PNG header decoder for image.DecodeConfig
	"math"
	"strings"

	"github.com/tidwall/gjson"
	"google.golang.org/protobuf/encoding/protowire"
)

// Byte-length token heuristics (~4 bytes per token) must not be applied to
// inline image data: base64 is ~33% larger than the binary image and the model
// bills images by pixel area, not by encoded size. A single 700x900 PNG
// screenshot is ~600 KB of base64 (~150k "tokens" by length) but costs ~900
// real input tokens. The helpers below charge inline images by pixel area.
const (
	// estimateBytesPerToken is the length heuristic used for non-image content.
	estimateBytesPerToken = 4
	// imageTokenPixelDivisor follows Anthropic's documented tokens ≈ w*h/750.
	imageTokenPixelDivisor = 750
	// imageTokenMaxLongEdge and imageTokenMaxPixels bound the server-side
	// resize. Values cover the high-resolution limits of current models so the
	// estimate errs on the generous (larger) side.
	imageTokenMaxLongEdge = 2576
	imageTokenMaxPixels   = 3_750_000
	// imageTokenFallback is charged when the image header cannot be decoded
	// (e.g. WebP or truncated data).
	imageTokenFallback int64 = 1600
	// imageTokenMinimum avoids charging near-zero for tiny icons.
	imageTokenMinimum int64 = 85
)

// EstimateImageTokens approximates the input tokens of one inline image from
// its base64 data (optionally a data: URL). Only the image header is decoded.
func EstimateImageTokens(base64Data string) int64 {
	data := strings.TrimSpace(base64Data)
	if strings.HasPrefix(data, "data:") {
		if idx := strings.IndexByte(data, ','); idx >= 0 {
			data = data[idx+1:]
		}
	}
	if data == "" {
		return 0
	}
	trimmed := strings.TrimRight(data, "=")
	if rem := len(trimmed) % 4; rem != 0 {
		trimmed += strings.Repeat("=", 4-rem)
	}
	decoder := base64.NewDecoder(base64.StdEncoding, strings.NewReader(trimmed))
	cfg, _, errDecode := image.DecodeConfig(decoder)
	if errDecode != nil || cfg.Width <= 0 || cfg.Height <= 0 {
		return imageTokenFallback
	}
	return imageTokensForDimensions(cfg.Width, cfg.Height)
}

func imageTokensForDimensions(width, height int) int64 {
	w := float64(width)
	h := float64(height)
	scale := 1.0
	if long := max(w, h); long > imageTokenMaxLongEdge {
		scale = imageTokenMaxLongEdge / long
	}
	if pixels := w * h * scale * scale; pixels > imageTokenMaxPixels {
		scale *= math.Sqrt(imageTokenMaxPixels / pixels)
	}
	tokens := int64(w * scale * h * scale / imageTokenPixelDivisor)
	if tokens < imageTokenMinimum {
		tokens = imageTokenMinimum
	}
	return tokens
}

// EstimateJSONPromptTokens applies the ~4 bytes/token heuristic to a JSON
// request body while charging inline images (base64 "data" with an image
// media type, or data:image/... URLs) by pixel area instead of encoded length.
func EstimateJSONPromptTokens(payload []byte) int64 {
	imageBytes, imageTokens := scanJSONInlineImages(gjson.ParseBytes(payload))
	textBytes := int64(len(payload)) - imageBytes
	if textBytes < 0 {
		textBytes = 0
	}
	return textBytes/estimateBytesPerToken + imageTokens
}

func scanJSONInlineImages(node gjson.Result) (imageBytes int64, imageTokens int64) {
	switch {
	case node.IsArray():
		node.ForEach(func(_, child gjson.Result) bool {
			b, t := scanJSONInlineImages(child)
			imageBytes += b
			imageTokens += t
			return true
		})
	case node.IsObject():
		mediaType := strings.ToLower(firstJSONString(node, "media_type", "mime_type", "mimeType"))
		node.ForEach(func(key, child gjson.Result) bool {
			if child.Type == gjson.String {
				value := child.String()
				if isInlineImageString(key.String(), value, mediaType) {
					imageBytes += int64(len(child.Raw))
					imageTokens += EstimateImageTokens(value)
				}
				return true
			}
			b, t := scanJSONInlineImages(child)
			imageBytes += b
			imageTokens += t
			return true
		})
	}
	return imageBytes, imageTokens
}

func firstJSONString(node gjson.Result, keys ...string) string {
	for _, key := range keys {
		if value := node.Get(key); value.Type == gjson.String {
			return value.String()
		}
	}
	return ""
}

func isInlineImageString(key, value, mediaType string) bool {
	if strings.HasPrefix(value, "data:image/") {
		return true
	}
	return key == "data" && strings.HasPrefix(mediaType, "image/") && len(value) > 0
}

// EstimateDevinWirePromptTokens estimates prompt tokens from a finalized Devin
// protobuf payload. Inline images (Prompt #10, data #1) are charged by pixel
// area; all other bytes keep the ~4 bytes/token heuristic.
func EstimateDevinWirePromptTokens(wire []byte) int64 {
	imageBytes, imageTokens := scanDevinWireImages(wire)
	textBytes := int64(len(wire)) - imageBytes
	if textBytes < 0 {
		textBytes = 0
	}
	return textBytes/estimateBytesPerToken + imageTokens
}

func scanDevinWireImages(wire []byte) (imageBytes int64, imageTokens int64) {
	forEachProtoBytesField(wire, func(num protowire.Number, prompt []byte) {
		if num != 3 {
			return
		}
		forEachProtoBytesField(prompt, func(num protowire.Number, img []byte) {
			if num != 10 {
				return
			}
			forEachProtoBytesField(img, func(num protowire.Number, data []byte) {
				if num != 1 || len(data) == 0 {
					return
				}
				imageBytes += int64(len(data))
				imageTokens += EstimateImageTokens(string(data))
			})
		})
	})
	return imageBytes, imageTokens
}

// forEachProtoBytesField calls fn for every length-delimited field in msg and
// stops silently on malformed input (the estimate then falls back to length).
func forEachProtoBytesField(msg []byte, fn func(protowire.Number, []byte)) {
	for len(msg) > 0 {
		num, kind, tagLen := protowire.ConsumeTag(msg)
		if tagLen < 0 {
			return
		}
		msg = msg[tagLen:]
		n := protowire.ConsumeFieldValue(num, kind, msg)
		if n < 0 {
			return
		}
		if kind == protowire.BytesType {
			if data, m := protowire.ConsumeBytes(msg); m >= 0 {
				fn(num, data)
			}
		}
		msg = msg[n:]
	}
}

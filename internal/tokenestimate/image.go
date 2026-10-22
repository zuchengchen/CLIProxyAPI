// Package tokenestimate provides byte-length prompt token heuristics that
// charge inline images by pixel area instead of encoded size. It has no
// internal dependencies so translators and executors can share it.
package tokenestimate

import (
	"encoding/base64"
	"image"
	_ "image/gif"  // register GIF header decoder for image.DecodeConfig
	_ "image/jpeg" // register JPEG header decoder for image.DecodeConfig
	_ "image/png"  // register PNG header decoder for image.DecodeConfig
	"math"
	"strings"

	"github.com/tidwall/gjson"
)

// Byte-length token heuristics (~4 bytes per token) must not be applied to
// inline image data: base64 is ~33% larger than the binary image and the model
// bills images by pixel area, not by encoded size. A single 700x900 PNG
// screenshot is ~600 KB of base64 (~150k "tokens" by length) but costs ~900
// real input tokens. The helpers below charge inline images by pixel area.
const (
	// BytesPerToken is the length heuristic used for non-image content.
	BytesPerToken = 4
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
	// InlineBlobTokens caps the estimate for one non-image inline base64 blob
	// (e.g. a PDF document); its encoded length says little about real cost.
	InlineBlobTokens int64 = 1600
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
	return estimateJSON(gjson.ParseBytes(payload), len(payload), false)
}

// EstimateJSONTokensWithBlobs is EstimateJSONPromptTokens for one raw JSON
// value that additionally charges every other inline base64 blob (data: URLs
// of any media type, base64 sources such as application/pdf documents, Gemini
// inlineData) at most InlineBlobTokens instead of its encoded length.
func EstimateJSONTokensWithBlobs(raw string) int64 {
	return estimateJSON(gjson.Parse(raw), len(raw), true)
}

func estimateJSON(node gjson.Result, size int, blobs bool) int64 {
	mediaBytes, mediaTokens := scanJSONInlineImages(node, blobs)
	textBytes := int64(size) - mediaBytes
	if textBytes < 0 {
		textBytes = 0
	}
	return textBytes/BytesPerToken + mediaTokens
}

func scanJSONInlineImages(node gjson.Result, blobs bool) (imageBytes int64, imageTokens int64) {
	switch {
	case node.IsArray():
		node.ForEach(func(_, child gjson.Result) bool {
			b, t := scanJSONInlineImages(child, blobs)
			imageBytes += b
			imageTokens += t
			return true
		})
	case node.IsObject():
		mediaType := strings.ToLower(firstJSONString(node, "media_type", "mime_type", "mimeType"))
		sourceType := node.Get("type").String()
		node.ForEach(func(key, child gjson.Result) bool {
			if child.Type == gjson.String {
				value := child.String()
				switch {
				case isInlineImageString(key.String(), value, mediaType):
					imageBytes += int64(len(child.Raw))
					imageTokens += EstimateImageTokens(value)
				case blobs && isInlineBlobString(key.String(), value, mediaType, sourceType):
					imageBytes += int64(len(child.Raw))
					imageTokens += min(int64(len(child.Raw))/BytesPerToken, InlineBlobTokens)
				}
				return true
			}
			b, t := scanJSONInlineImages(child, blobs)
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

// isInlineBlobString reports base64 payloads that are not images: data: URLs
// with a ;base64 header, Anthropic {"type":"base64","data":...} sources, and
// Gemini inlineData {"mimeType":...,"data":...}. Plain-text document sources
// ({"type":"text","data":...}) stay on the length heuristic.
func isInlineBlobString(key, value, mediaType, sourceType string) bool {
	if value == "" {
		return false
	}
	if strings.HasPrefix(value, "data:") {
		header, _, found := strings.Cut(value, ",")
		return found && strings.HasSuffix(strings.ToLower(header), ";base64")
	}
	return key == "data" && (sourceType == "base64" || (sourceType == "" && mediaType != ""))
}

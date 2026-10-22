package helps

import (
	"github.com/router-for-me/CLIProxyAPI/v8/internal/tokenestimate"
	"google.golang.org/protobuf/encoding/protowire"
)

// EstimateDevinWirePromptTokens estimates prompt tokens from a finalized Devin
// protobuf payload. Inline images (Prompt #10, data #1) are charged by pixel
// area; all other bytes keep the ~4 bytes/token heuristic.
func EstimateDevinWirePromptTokens(wire []byte) int64 {
	imageBytes, imageTokens := scanDevinWireImages(wire)
	textBytes := int64(len(wire)) - imageBytes
	if textBytes < 0 {
		textBytes = 0
	}
	return textBytes/tokenestimate.BytesPerToken + imageTokens
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
				imageTokens += tokenestimate.EstimateImageTokens(string(data))
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

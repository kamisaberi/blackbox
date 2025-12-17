package utils

import (
	"bytes"
	"encoding/json"
	"strings"
)

// MinifyJSON takes raw bytes, ensures valid JSON, and removes all newlines/spaces.
// If input is not JSON, it treats it as a raw string and escapes it.
func MinifyJSON(input []byte) string {
	var dst bytes.Buffer
	
	// Try to compact the JSON (removes \n and whitespace)
	err := json.Compact(&dst, input)
	if err != nil {
		// Not valid JSON? Treat as raw string.
		// Escape newlines so it fits on one TCP line.
		raw := string(input)
		raw = strings.ReplaceAll(raw, "\n", "\\n")
		raw = strings.ReplaceAll(raw, "\r", "")
		return raw
	}

	return dst.String()
}
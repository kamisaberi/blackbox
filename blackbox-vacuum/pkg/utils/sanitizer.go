package utils

import (
	"regexp"
)

// Pre-compile regex for performance
var (
	// Looks for standard AWS Key patterns (AKIA...)
	awsKeyRegex = regexp.MustCompile(`(AKIA|ASIA)[0-9A-Z]{16}`)
	
	// Looks for generic "password": "..." patterns in JSON
	jsonPassRegex = regexp.MustCompile(`"(password|secret|token|key)":\s*"([^"]+)"`)
	
	// Looks for standard private key headers
	privateKeyRegex = regexp.MustCompile(`-----BEGIN (RSA|EC|DSA|OPENSSH) PRIVATE KEY-----`)
)

// Sanitize takes a raw string and masks sensitive data
func Sanitize(input string) string {
	cleaned := input

	// 1. Mask AWS Keys
	cleaned = awsKeyRegex.ReplaceAllString(cleaned, "[REDACTED_AWS_KEY]")

	// 2. Mask JSON secrets (e.g. "password": "hunter2" -> "password": "[REDACTED]")
	cleaned = jsonPassRegex.ReplaceAllString(cleaned, `"$1": "[REDACTED]"`)

	// 3. Mask Private Keys
	if privateKeyRegex.MatchString(cleaned) {
		return "[REDACTED_PRIVATE_KEY_BLOCK]"
	}

	return cleaned
}

// SanitizeBytes is a helper for byte slices
func SanitizeBytes(input []byte) []byte {
	return []byte(Sanitize(string(input)))
}
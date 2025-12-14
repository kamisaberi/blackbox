#ifndef SENTRY_H
#define SENTRY_H

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include "config.h"

// Simple Logging Macros (Map to printf/stderr)
// In a real RTOS, map these to UART or RTT
#define LOG_INFO(fmt, ...) fprintf(stdout, "[INF] " fmt "\n", ##__VA_ARGS__)
#define LOG_ERR(fmt, ...)  fprintf(stderr, "[ERR] " fmt "\n", ##__VA_ARGS__)
#define LOG_DBG(fmt, ...)  fprintf(stdout, "[DBG] " fmt "\n", ##__VA_ARGS__)

#endif // SENTRY_H
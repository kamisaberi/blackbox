// src/hal/linux/sys_posix.c
#include "hal/sys.h"
#include <unistd.h>
#include <string.h>

void hal_sys_get_device_id(char* buffer, size_t max_len) {
    // Try to get hostname
    if (gethostname(buffer, max_len) != 0) {
        // Fallback
        strncpy(buffer, "UNKNOWN-DEVICE", max_len);
    }
    // Ensure null termination
    buffer[max_len - 1] = '\0';
}
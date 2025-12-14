// src/hal/linux/time_posix.c
#include "hal/time.h"
#include <time.h>
#include <unistd.h>

uint64_t hal_get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)(ts.tv_sec * 1000 + ts.tv_nsec / 1000000);
}

void hal_sleep_ms(uint32_t ms) {
    usleep(ms * 1000); // usleep takes microseconds
}
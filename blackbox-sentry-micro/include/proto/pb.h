#ifndef PB_H_MINIMAL
#define PB_H_MINIMAL
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

typedef struct pb_ostream_s pb_ostream_t;
typedef struct pb_istream_s pb_istream_t;
typedef struct pb_msgdesc_s pb_msgdesc_t;
typedef struct pb_callback_s {
    void* funcs;
    void* arg;
} pb_callback_t;

// Minimal stream setup
pb_ostream_t pb_ostream_from_buffer(uint8_t *buf, size_t bufsize);
bool pb_encode(pb_ostream_t *stream, const pb_msgdesc_t *fields, const void *src_struct);

#endif
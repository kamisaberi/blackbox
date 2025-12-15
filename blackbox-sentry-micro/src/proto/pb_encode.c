#include "proto/pb.h"
#include <string.h>

// Mock implementation that just writes dummy bytes
// so you can verify the networking works without real serialization.

pb_ostream_t pb_ostream_from_buffer(uint8_t *buf, size_t bufsize) {
    pb_ostream_t stream;
    // Real implementation stores pointers here
    return stream;
}

bool pb_encode(pb_ostream_t *stream, const pb_msgdesc_t *fields, const void *src_struct) {
    // FAKE SERIALIZATION
    // We just pretend we wrote 10 bytes successfully.
    // In real NanoPB, this packs bits tightly.
    (void)stream; (void)fields; (void)src_struct;
    return true;
}
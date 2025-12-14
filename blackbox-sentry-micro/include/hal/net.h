#ifndef HAL_NET_H
#define HAL_NET_H

#include <stddef.h>
#include <stdint.h>

// Opaque handle
typedef struct net_socket_t net_socket_t;

// Connect to remote host
int hal_net_connect(net_socket_t* sock, const char* ip, uint16_t port);

// Send bytes
int hal_net_send(net_socket_t* sock, const uint8_t* data, size_t len);

// Close
void hal_net_close(net_socket_t* sock);

#endif
#include "hal/net.h"
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>

// Concrete definition of the opaque struct
struct net_socket_t {
    int fd;
};

int hal_net_connect(net_socket_t* sock, const char* ip, uint16_t port) {
    sock->fd = socket(AF_INET, SOCK_STREAM, 0);
    if (sock->fd < 0) return -1;

    struct sockaddr_in serv_addr;
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(port);

    if (inet_pton(AF_INET, ip, &serv_addr.sin_addr) <= 0) return -2;

    if (connect(sock->fd, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        return -3;
    }
    return 0; // Success
}

int hal_net_send(net_socket_t* sock, const uint8_t* data, size_t len) {
    return send(sock->fd, data, len, 0);
}
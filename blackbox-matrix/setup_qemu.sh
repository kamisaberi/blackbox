#!/bin/bash
echo ">>> Enabling ARM Emulation for Docker..."
docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
echo ">>> Done. You can now run ARM32 and ARM64 containers."
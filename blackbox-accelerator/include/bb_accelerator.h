#ifndef BB_ACCELERATOR_H
#define BB_ACCELERATOR_H

#include <string>
#include <vector>
#include <cstdint>

namespace bb_accel {

    class Accelerator {
    public:
        Accelerator();
        ~Accelerator();

        /**
         * @brief Initialize PCIe and load Firmware.
         * @param xclbin_path Path to the compiled hardware bitstream.
         * @return 0 on success.
         */
        int init(const std::string& xclbin_path);

        /**
         * @brief Push a list of IPs to block to the FPGA memory.
         * The FPGA reads this memory directly.
         */
        void update_firewall_rules(const std::vector<uint32_t>& bad_ips);

        /**
         * @brief Poll the FPGA for processed packets.
         * Uses Zero-Copy DMA.
         */
        int poll_processed_data(char* buffer, size_t max_len);

    private:
        // Pimpl idiom to hide Xilinx/OpenCL dependencies from the Core
        class Impl;
        Impl* pImpl;
    };

}

#endif
/**
 * @file packet_filter.cpp
 * @brief FPGA Kernel Logic.
 * 
 * This function is synthesized into an electrical circuit.
 * It pipelines data 64-bytes per clock cycle.
 */

#include <hls_stream.h>
#include <ap_int.h>
#include <cstdint>

// Define 512-bit data width (standard for 100Gbps Ethernet)
typedef ap_uint<512> axis_data;
typedef ap_uint<1>   axis_keep;

struct AxisWord {
    axis_data data;
    axis_keep keep;
    bool last;
};

// Hardware Function
void blackbox_kernel(hls::stream<AxisWord>& input_stream, 
                     hls::stream<AxisWord>& output_stream,
                     unsigned int bad_ip_list[1000]) 
{
    #pragma HLS INTERFACE axis port=input_stream
    #pragma HLS INTERFACE axis port=output_stream
    #pragma HLS INTERFACE m_axi port=bad_ip_list bundle=gmem

    AxisWord packet;
    
    // PIPELINE directive allows processing 1 packet per clock cycle
    while (input_stream.read(packet)) {
        #pragma HLS PIPELINE II=1
        
        // 1. Extract IP Header (Mock logic for 512-bit slice)
        // In real HLS, we slice bits: packet.data.range(159, 128)
        unsigned int src_ip = packet.data.range(255, 224); // Example offset

        // 2. Hardware Firewall Check
        bool drop = false;
        
        // Unroll loop for parallel checking in hardware
        for (int i = 0; i < 1000; ++i) {
            #pragma HLS UNROLL factor=16
            if (bad_ip_list[i] == src_ip) {
                drop = true;
            }
        }

        // 3. Forward or Drop
        if (!drop) {
            output_stream.write(packet);
        }
        
        if (packet.last) break;
    }
}
/**
 * @file crash_handler.cpp
 * @brief Implementation of Stack Unwinding.
 */

#include "blackbox/common/crash_handler.h"
#include <iostream>
#include <csignal>
#include <cstdlib>
#include <cstring>    // for strsignal
#include <execinfo.h> // Linux/POSIX specific for backtrace
#include <unistd.h>

namespace blackbox::common {

    // =========================================================
    // Install Handlers
    // =========================================================
    void CrashHandler::install() {
        std::signal(SIGSEGV, handle_crash); // Invalid memory access (Segfault)
        std::signal(SIGABRT, handle_crash); // Abort signal (assert failure)
        std::signal(SIGFPE,  handle_crash); // Floating point exception (divide by zero)
        std::signal(SIGILL,  handle_crash); // Illegal instruction
    }

    // =========================================================
    // Crash Callback
    // =========================================================
    void CrashHandler::handle_crash(int signal) {
        // Use std::cerr because it is unbuffered. std::cout might not flush in a crash.
        std::cerr << "\n\n" 
                  << "########################################################\n"
                  << "   FATAL ERROR: Blackbox Core Crashed!\n"
                  << "   Signal: " << signal << " (" << strsignal(signal) << ")\n"
                  << "########################################################\n";

        print_stack_trace();

        std::cerr << "########################################################\n"
                  << "   Terminating immediately.\n"
                  << "########################################################\n";

        // Exit with failure code so K8s knows to restart the pod
        std::exit(EXIT_FAILURE);
    }

    // =========================================================
    // Stack Trace Printer
    // =========================================================
    void CrashHandler::print_stack_trace() {
        const int MAX_FRAMES = 64;
        void* addrlist[MAX_FRAMES];

        // Retrieve current stack addresses
        int addrlen = backtrace(addrlist, sizeof(addrlist) / sizeof(void*));

        if (addrlen == 0) {
            std::cerr << "  <empty, possibly corrupt stack>" << std::endl;
            return;
        }

        // Resolve addresses into symbols (mangled function names)
        // Note: To see Demangled names (e.g. Class::Method), run the output through 'c++filt'
        char** symbollist = backtrace_symbols(addrlist, addrlen);

        for (int i = 0; i < addrlen; i++) {
            std::cerr << "  [" << i << "] " << (symbollist[i] ? symbollist[i] : "<unknown>") << std::endl;
        }

        // backtrace_symbols allocates memory via malloc, we must free it.
        // Although we are crashing, it's good practice.
        std::free(symbollist);
    }

} // namespace blackbox::common
/**
 * @file crash_handler.h
 * @brief Fatal Error Interceptor & Stack Tracer.
 * 
 * Captures segmentation faults and illegal instructions.
 * Prints a human-readable stack trace to stderr to aid debugging.
 * Essential for C++ applications running in containers.
 */

#ifndef BLACKBOX_COMMON_CRASH_HANDLER_H
#define BLACKBOX_COMMON_CRASH_HANDLER_H

namespace blackbox::common {

    class CrashHandler {
    public:
        /**
         * @brief Installs signal handlers for SIGSEGV, SIGILL, SIGFPE, SIGABRT.
         * Call this at the very beginning of main().
         */
        static void install();

    private:
        /**
         * @brief The callback triggered by the OS when the app crashes.
         * @param signal The signal number (e.g., 11 for SIGSEGV)
         */
        static void handle_crash(int signal);

        /**
         * @brief Captures and prints the current call stack using 'execinfo'.
         */
        static void print_stack_trace();
    };

} // namespace blackbox::common

#endif // BLACKBOX_COMMON_CRASH_HANDLER_H
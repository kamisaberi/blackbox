/**
 * @file application.h
 * @brief The Main Entry Wrapper.
 * 
 * Encapsulates the startup and shutdown sequence of the Blackbox Engine.
 * Ensures strict ordering of component initialization.
 */

#ifndef BLACKBOX_CORE_APPLICATION_H
#define BLACKBOX_CORE_APPLICATION_H

#include <memory>
#include "blackbox/core/pipeline.h"

namespace blackbox::core {

    class Application {
    public:
        /**
         * @brief Construct the Application.
         * Parses command line args (if any) and loads settings.
         */
        Application(int argc, char** argv);
        ~Application();

        /**
         * @brief Starts the engine and blocks until an OS signal (Ctrl+C) is received.
         * @return int Exit code (0 = Success, 1 = Error)
         */
        int run();

    private:
        /**
         * @brief Helper to print the ASCII banner on startup.
         */
        void print_banner();

        /**
         * @brief Initialize static subsystems (Logger, CrashHandler).
         */
        void init_subsystems();

        // The main orchestration engine
        std::unique_ptr<Pipeline> pipeline_;
    };

} // namespace blackbox::core

#endif // BLACKBOX_CORE_APPLICATION_H
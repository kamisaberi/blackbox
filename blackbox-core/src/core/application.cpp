/**
 * @file application.cpp
 * @brief Implementation of the Bootstrap Sequence.
 */

#include "blackbox/core/application.h"
#include "blackbox/common/settings.h"
#include "blackbox/common/logger.h"
#include "blackbox/common/signal_handler.h"
#include "blackbox/common/crash_handler.h"
#include "blackbox/common/metrics.h"
#include <iostream>
#include <stdexcept>

namespace blackbox::core {

    // =========================================================
    // Constructor
    // =========================================================
    Application::Application(int argc, char** argv) {
        // 1. Initialize Safety Nets first
        common::CrashHandler::install();

        // 2. Initialize Subsystems
        init_subsystems();

        // 3. Load Configuration
        // (In the future, parse argc/argv for config file path)
        common::Settings::instance().load_from_env();
    }

    // =========================================================
    // Destructor
    // =========================================================
    Application::~Application() {
        // Pipeline is destroyed automatically via unique_ptr
        // Logger shutdown if necessary
    }

    // =========================================================
    // Subsystem Init
    // =========================================================
    void Application::init_subsystems() {
        // Set default log level
        common::Logger::instance().set_level(common::LogLevel::INFO);
        
        // Register Signal Handlers (SIGINT, SIGTERM)
        common::SignalHandler::instance().register_handlers();
    }

    // =========================================================
    // Banner
    // =========================================================
    void Application::print_banner() {
        std::cout << "\n"
                  << "   ██████╗ ██╗      █████╗  ██████╗██╗  ██╗██████╗  ██████╗ ██╗  ██╗\n"
                  << "   ██╔══██╗██║     ██╔══██╗██╔════╝██║ ██╔╝██╔══██╗██╔═══██╗╚██╗██╔╝\n"
                  << "   ██████╔╝██║     ███████║██║     █████╔╝ ██████╔╝██║   ██║ ╚███╔╝ \n"
                  << "   ██╔══██╗██║     ██╔══██║██║     ██╔═██╗ ██╔══██╗██║   ██║ ██╔██╗ \n"
                  << "   ██████╔╝███████╗██║  ██║╚██████╗██║  ██╗██████╔╝╚██████╔╝██╔╝ ██╗\n"
                  << "   ╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚═════╝  ╚═════╝ ╚═╝  ╚═╝\n"
                  << "   :: Kinetic Defense Engine :: v0.1.0 ::\n\n";
    }

    // =========================================================
    // Run (The Main Loop)
    // =========================================================
    int Application::run() {
        print_banner();

        try {
            // 1. Instantiate the Pipeline
            // This initializes Network, AI, and Storage
            pipeline_ = std::make_unique<Pipeline>();

            // 2. Start the Threads
            pipeline_->start();

            // 3. Start Metrics Reporting (Background)
            common::Metrics::instance().start_reporter(5);

            // 4. BLOCK MAIN THREAD
            // Wait here until Ctrl+C is pressed
            common::SignalHandler::instance().wait_for_signal();

            // 5. Shutdown Sequence
            // (Reached after SignalHandler releases the lock)
            LOG_WARN("Shutdown sequence initiated...");
            
            common::Metrics::instance().stop();
            pipeline_->stop();

            LOG_INFO("Blackbox shutdown complete.");
            return 0;

        } catch (const std::exception& e) {
            LOG_CRITICAL("Unhandled exception during runtime: " + std::string(e.what()));
            return 1;
        } catch (...) {
            LOG_CRITICAL("Unknown fatal error occurred.");
            return 1;
        }
    }

} // namespace blackbox::core
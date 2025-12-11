/**
 * @file geoip_service.cpp
 * @brief Implementation of MaxMind DB Lookups.
 */

#include "blackbox/enrichment/geoip_service.h"
#include "blackbox/common/logger.h"
#include <iostream>

namespace blackbox::enrichment {

    // =========================================================
    // Constructor
    // =========================================================
    GeoIPService::GeoIPService(const std::string& db_path) {
        LOG_INFO("Loading GeoIP Database from: " + db_path);

        int status = MMDB_open(db_path.c_str(), MMDB_MODE_MMAP, &mmdb_);

        if (status != MMDB_SUCCESS) {
            std::string err = "Failed to open GeoIP DB: " + std::string(MMDB_strerror(status));
            LOG_ERROR(err);
            ready_ = false;
            // We don't throw here to allow the app to start even if GeoIP fails
            // (Graceful degradation)
        } else {
            ready_ = true;
            LOG_INFO("GeoIP Service Ready.");
        }
    }

    // =========================================================
    // Destructor
    // =========================================================
    GeoIPService::~GeoIPService() {
        if (ready_) {
            MMDB_close(&mmdb_);
        }
    }

    // =========================================================
    // Helper: Extract String from MMDB Entry
    // =========================================================
    static std::string get_mmdb_string(MMDB_entry_data_s* entry_data) {
        if (entry_data->has_data && entry_data->type == MMDB_DATA_TYPE_UTF8_STRING) {
            return std::string(entry_data->utf8_string, entry_data->data_size);
        }
        return "";
    }

    // =========================================================
    // Lookup (The Hot Path)
    // =========================================================
    std::optional<GeoLocation> GeoIPService::lookup(std::string_view ip_address) {
        if (!ready_) return std::nullopt;

        int gai_error, mmdb_error;
        std::string ip_str(ip_address); // MMDB API requires null-terminated string

        MMDB_lookup_result_s result = MMDB_lookup_string(
            &mmdb_, 
            ip_str.c_str(), 
            &gai_error, 
            &mmdb_error
        );

        if (gai_error != 0 || mmdb_error != MMDB_SUCCESS || !result.found_entry) {
            // Private IP (LAN) or Invalid
            return std::nullopt;
        }

        GeoLocation loc;
        MMDB_entry_data_s entry_data;

        // 1. Get Country Code (ISO)
        // Path: country -> iso_code
        if (MMDB_get_value(&result.entry, &entry_data, "country", "iso_code", NULL) == MMDB_SUCCESS) {
            loc.country_iso = get_mmdb_string(&entry_data);
        }

        // 2. Get City Name
        // Path: city -> names -> en
        if (MMDB_get_value(&result.entry, &entry_data, "city", "names", "en", NULL) == MMDB_SUCCESS) {
            loc.city = get_mmdb_string(&entry_data);
        }

        // 3. Get Latitude
        // Path: location -> latitude
        if (MMDB_get_value(&result.entry, &entry_data, "location", "latitude", NULL) == MMDB_SUCCESS) {
            if (entry_data.has_data && entry_data.type == MMDB_DATA_TYPE_DOUBLE) {
                loc.latitude = entry_data.double_value;
            }
        }

        // 4. Get Longitude
        // Path: location -> longitude
        if (MMDB_get_value(&result.entry, &entry_data, "location", "longitude", NULL) == MMDB_SUCCESS) {
            if (entry_data.has_data && entry_data.type == MMDB_DATA_TYPE_DOUBLE) {
                loc.longitude = entry_data.double_value;
            }
        }

        return loc;
    }

} // namespace blackbox::enrichment
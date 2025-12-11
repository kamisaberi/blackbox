/**
 * @file geoip_service.h
 * @brief High-performance IP Geolocation.
 * 
 * Uses libmaxminddb (MMDB) for local, zero-latency lookups.
 * Required for the "Threat Map" visualization.
 */

#ifndef BLACKBOX_ENRICHMENT_GEOIP_SERVICE_H
#define BLACKBOX_ENRICHMENT_GEOIP_SERVICE_H

#include <string>
#include <string_view>
#include <optional>
#include <maxminddb.h> // Requires libmaxminddb-dev

namespace blackbox::enrichment {

    struct GeoLocation {
        std::string country_iso; // "US", "CN", "RU"
        std::string city;        // "New York"
        double latitude = 0.0;
        double longitude = 0.0;
    };

    class GeoIPService {
    public:
        /**
         * @brief Initialize with path to GeoLite2-City.mmdb
         */
        explicit GeoIPService(const std::string& db_path);
        ~GeoIPService();

        /**
         * @brief Lookup IP location.
         * 
         * @param ip_address IPv4 or IPv6 string
         * @return std::optional<GeoLocation> Data if found, nullopt if private/invalid
         */
        std::optional<GeoLocation> lookup(std::string_view ip_address);

    private:
        MMDB_s mmdb_;
        bool ready_ = false;
    };

} // namespace blackbox::enrichment

#endif // BLACKBOX_ENRICHMENT_GEOIP_SERVICE_H
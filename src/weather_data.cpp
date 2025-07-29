#include "weather_data.hpp"
#include <iostream>

bool WeatherData::load(const char* path) {
    std::cout << "Loading weather data from " << path << std::endl;
    return true;
}

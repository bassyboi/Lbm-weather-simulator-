#include "weather_data.hpp"
#include <iostream>

int main() {
    WeatherData data;
    data.load("forecast.nc");
    std::cout << "Running weather forecast..." << std::endl;
    return 0;
}

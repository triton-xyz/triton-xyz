#include "Device.h"

#include <stdexcept>

namespace proton {

namespace {

Device makeSyntheticDevice(DeviceType type, uint64_t index,
                           const char *archName) {
  return Device(type, index, 0, 0, 0, 0, archName);
}

} // namespace

Device getDevice(DeviceType type, uint64_t index) {
  switch (type) {
  case DeviceType::HIP:
    return makeSyntheticDevice(type, index, "hip");
  case DeviceType::CUDA:
    return makeSyntheticDevice(type, index, "cuda");
  case DeviceType::CPU:
    return makeSyntheticDevice(type, index, "cpu");
  case DeviceType::COUNT:
    break;
  }
  throw std::runtime_error("DeviceType not supported");
}

const std::string getDeviceTypeString(DeviceType type) {
  if (type == DeviceType::CUDA) {
    return DeviceTraits<DeviceType::CUDA>::name;
  }
  if (type == DeviceType::HIP) {
    return DeviceTraits<DeviceType::HIP>::name;
  }
  if (type == DeviceType::CPU) {
    return DeviceTraits<DeviceType::CPU>::name;
  }
  throw std::runtime_error("DeviceType not supported");
}

} // namespace proton

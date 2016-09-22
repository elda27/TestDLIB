#include "ReadMNISTDatas.hpp"
#include <fstream>
#include <iostream>

bool readLabels(std::string const& filename, std::vector<int>& labels)
{
  std::fstream input(filename, std::ios::in | std::ios::binary);

  const std::uint32_t MAGIC_NUMBER = 0x00000801;
  std::uint32_t magic_number = 0, n_labels = 0;

  input.read(reinterpret_cast<char*>(&magic_number), 4);
  input.read(reinterpret_cast<char*>(&n_labels), 4);

  magic_number = _byteswap_ulong(magic_number);
  n_labels = _byteswap_ulong(n_labels);

  if (magic_number != MAGIC_NUMBER)
  {
    std::cerr << "Wrong magic number." << __FUNCTION__ << std::endl;
    return false;
  }

  labels.reserve(n_labels);
  for (std::size_t i = 0; i < n_labels; ++i)
  {
    std::uint8_t label = 0;
    input.read(reinterpret_cast<char*>(&label), 1);
    labels.emplace_back(label);
  }
}
bool readImages(std::string const& filename, std::vector<cv::Mat>& images)
{
  std::fstream input(filename, std::ios::in | std::ios::binary);

  const std::uint32_t MAGIC_NUMBER = 0x00000803;
  std::uint32_t magic_number = 0, n_images = 0, image_width = 0, image_height = 0;


  input.read(reinterpret_cast<char*>(&magic_number), 4);
  input.read(reinterpret_cast<char*>(&n_images), 4);
  input.read(reinterpret_cast<char*>(&image_width), 4);
  input.read(reinterpret_cast<char*>(&image_height), 4);

  magic_number = _byteswap_ulong(magic_number);
  n_images = _byteswap_ulong(n_images);
  image_width = _byteswap_ulong(image_width);
  image_height = _byteswap_ulong(image_height);

  if (magic_number != MAGIC_NUMBER)
  {
    std::cerr << "Wrong magic number." << __FUNCTION__ << std::endl;
    return false;
  }

  images.reserve(n_images);
  for (std::size_t i = 0; i < n_images; ++i)
  {
    cv::Mat image = cv::Mat::zeros(image_width, image_height, CV_8UC1);
    input.read(reinterpret_cast<char*>(image.data), image_width * image_height);
    images.emplace_back(image);
  }
}
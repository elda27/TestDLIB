#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <map>
#include <cstdint>
#include <filesystem>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include "ReadMNISTDatas.hpp"

bool saveImage(std::vector<cv::Mat> const& images, std::vector<int> const& labels, std::string const& output_dir);


int main(int argc, char** argv)
{
  if (argc < 4)
  {
    std::cerr << "Wrong arguments\nConvertLabelAndImage <images> <labels> <output directory>" << std::endl;
    return -1;
  }
  const std::string IMAGE_FILENAME = argv[1];
  const std::string LABEL_FILENAME = argv[2];
  const std::string OUTPUT_DIRECTORY = argv[3];

  namespace fs = std::experimental::filesystem;
  if (!fs::exists(OUTPUT_DIRECTORY))
  {
    fs::create_directories(OUTPUT_DIRECTORY);
  }

  std::vector<cv::Mat> images;
  std::vector<int> labels;

  if (!readImages(IMAGE_FILENAME, images))
  {
    return -1;
  }
  if (!readLabels(LABEL_FILENAME, labels))
  {
    return -1;
  }

  saveImage(images, labels, OUTPUT_DIRECTORY);

  return 0;
}

bool saveImage(std::vector<cv::Mat> const& images, std::vector<int> const& labels, std::string const& output_dir)
{
  if (images.size() != labels.size())
  {
    std::cerr << "Not compare between images.size and labels.size." << std::endl;
    return false;
  }

  std::map<int, std::uint32_t> counter;
  using Path = std::experimental::filesystem::path;
  Path output_directory(output_dir);

  for (std::size_t i = 0; i < images.size(); ++i)
  {
    std::ostringstream oss;
    oss << labels[i] << "_" << counter[labels[i]]++ << ".png";
    cv::imwrite((output_directory / oss.str()).string(), images[i]);
  }

  return true;
}

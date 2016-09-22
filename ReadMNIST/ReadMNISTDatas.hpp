#ifndef LOAD_MNIST_DATAS_HPP
#define LOAD_MNIST_DATAS_HPP

#include <vector>
#include <string>
#include <opencv2/core.hpp>

bool readLabels(std::string const& filename, std::vector<int>& labels);
bool readImages(std::string const& filename, std::vector<cv::Mat>& images);

#endif //!LOAD_MNIST_DATAS_HPP

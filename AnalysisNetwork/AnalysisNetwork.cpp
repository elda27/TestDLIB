#include <iostream>
#include <fstream>
#include <chrono>

#include <opencv2/core.hpp>

#include <dlib/opencv.h>
#include <dlib/data_io.h>
#include <dlib/dnn.h>
#include <dlib/array2d.h>
#include <dlib/image_transforms.h>

#include "ReadMNISTDatas.hpp"
#include "TrainFilePath.hpp"
#include "../TestDNN/DLIBInput.hpp"
#include "NetworkStringTraits.hpp"

namespace enabler { extern void* enabler; }

template <class T>
using UnqualifiedType = std::remove_pointer_t<std::remove_cv_t<std::remove_reference_t<T>>>;

template <class Network>
void saveWeights(std::ostream& output, Network&& net);

template <class Network, class T>
void saveWeightsImpl(std::ostream& output, Network&& net, dlib::input<T> const& input);

template <class Network, class Subnet>
void saveWeightsImpl(std::ostream& output, Network&& net, Subnet&& subnet);

int main()
{
  using net_type = dlib::loss_multiclass_log<
    dlib::fc<10,
    dlib::relu<dlib::fc<84,
    dlib::relu<dlib::fc<120,
    dlib::max_pool<2, 2, 2, 2, dlib::relu<dlib::con<16, 5, 5, 1, 1,
    dlib::max_pool<2, 2, 2, 2, dlib::relu<dlib::con<6, 5, 5, 1, 1,
    dlib::input<cv::Mat>
    >>>>>>>>>>>>;

  net_type net;
  dlib::deserialize("mnist_network.dat") >> net;

  std::fstream output("weights.csv", std::ios::out | std::ios::trunc);
  saveWeights(output, net);

  return 0;
}

template <class Network>
void saveWeights(std::ostream& output, Network&& net)
{
  auto&& subnet = net.subnet();
  saveWeightsImpl(output, subnet, subnet.subnet());
}

using InputLayer = dlib::input<cv::Mat>;
/*
template <
  class Network, 
  class Subnet, 
  std::enable_if_t<
    std::is_same<UnqualifiedType<Subnet>, dlib::input<cv::Mat>>::value
  >*& = enabler::enabler
>
*/
template <class Network>
void saveWeightsImpl(std::ostream& output, Network&& net, InputLayer input)
{
  return;
}

template <class Network, class Subnet>
void saveWeightsImpl(std::ostream& output, Network&& net, Subnet&& subnet)
{
  using NetworkType = std::remove_reference_t<Network>;
  using SubnetType = std::remove_reference_t<Subnet>;
  saveWeightsImpl(
    output, 
    std::forward<NetworkType::subnet_type>(net.subnet()),
    std::forward<SubnetType::subnet_type>(subnet.subnet())
  );

  output << NetworkStringTraits<NetworkType>::STRING << "\n";
  dlib::tensor const& tensor = net.layer_details().get_layer_params();
  output << "Layer:" << NetworkType::num_layers << ","
         << "Row:" << tensor.nr() << ","
         << "Column:" << tensor.nc() << ","
         << "Channel:" << tensor.k() << "\n";

  for (long k = 0; k < tensor.k(); ++k)
  {
    auto ptr = tensor.host();
    output << "Node" << k << "\n";
    for (long r = 0; r < tensor.nr(); ++r)
    {
      for (long c = 0; c < tensor.nc(); ++c)
      {
        output << ptr[c * tensor.k() + r * tensor.nc() + k] << ",";
      }
      output.seekp(std::ios::cur, -1) << "\n";
    }
    output << "\n";
  }

  output << "\n";
}

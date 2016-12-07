#ifndef NETWORK_STRING_HPP
#define NETWORK_STRING_HPP

#include <string>
#include <dlib/dnn.h>

template <class Subnet>
struct NetworkStringTraits
{
};

template <class Subnet>
struct NetworkStringTraits<dlib::relu<Subnet>>
{
  static constexpr char const* STRING = "ReLU";
};

template <class Layer, class Subnet>
struct NetworkStringTraits<dlib::add_layer<Layer, Subnet>>
{
  static constexpr char const* STRING = NetworkStringTraits<Subnet>::STRING;
};

template <long n, class Subnet>
struct NetworkStringTraits<dlib::fc<n, Subnet>>
{
  static constexpr char const* STRING = "FullyConnectedLayer";
};

template <long nr, long nc, long sy, long sx, class Subnet>
struct NetworkStringTraits<dlib::max_pool<nr, nc, sy, sx, Subnet>>
{
  static constexpr char const* STRING = "MaxPooling";
};

template <long nf, long nr, long nc, int sy, int sx, typename Subnet>
struct NetworkStringTraits<dlib::con<nf, nr, nc, sy, sx, Subnet>>
{
  static constexpr char const* STRING = "Convolution";
};

#endif //!NETWORK_STRING_HPP
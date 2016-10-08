#ifndef DLIB_INPUT_HPP
#define DLIB_INPUT_HPP

#include <dlib/dnn/tensor.h>
#include <dlib/dnn/input.h>
#include <dlib/opencv/cv_image.h>
#include <limits>

namespace dlib
{
  template <>
  class input<cv::Mat>
  {
  public:
    using input_type = cv::Mat;
    const static unsigned int sample_expansion_factor = 1;

    template <typename input_iterator>
    void to_tensor(
      input_iterator begin,
      input_iterator end,
      resizable_tensor& data
    ) const
    {
      DLIB_CASSERT(std::distance(begin, end) > 0, "");
      
      switch (begin->depth())
      {
      case CV_8S:
        to_tensor_<char>(begin, end, data);
        break;
      case CV_8U:
        to_tensor_<uchar>(begin, end, data);
        break;
      case CV_16S:
        to_tensor_<short>(begin, end, data);
        break;
      case CV_16U:
        to_tensor_<unsigned short>(begin, end, data);
        break;
      case CV_32S:
        to_tensor_<int>(begin, end, data);
        break;
      case CV_32F:
        to_tensor_<float>(begin, end, data);
        break;
      case CV_64F:
        to_tensor_<double>(begin, end, data);
        break;
      default:
        throw;
      }
    }

    friend void serialize(const input& item, std::ostream& out)
    {
      serialize("input<cv::Mat>", out);
    }

    friend void deserialize(input& item, std::istream& in)
    {
      std::string version;
      deserialize(version, in);
      if (version != "input<cv::Mat>")
        throw serialization_error("Unexpected version found while deserializing dlib::input.");
    }

    friend std::ostream& operator<<(std::ostream& out, const input& item)
    {
      out << "input<cv::Mat>";
      return out;
    }

    friend void to_xml(const input& item, std::ostream& out)
    {
      out << "<input/>";
    }

  private:
    /**
  	* \brief enumerable<cv::Mat> のイテレータをtensor型に変換する．
  	* \param [in] begin
  	* \param [in] end
  	*/
    template <typename Tp, typename input_iterator>
    void to_tensor_(
      input_iterator begin,
      input_iterator end,
      resizable_tensor& data
    ) const
    {
      using value_type = Tp;
      cv::Mat mat = *begin;
      
      // initialize data to the right size to contain the stuff in the iterator range.
      data.set_size(std::distance(begin, end), mat.channels(), mat.rows, mat.cols);

      const size_t offset = mat.step * mat.rows;
      auto ptr = data.host();
      for (auto it = begin; it != end; ++it)
      {
        for (int y = 0; y < mat.rows; ++y)
        {
          for (int x = 0; x < mat.cols; ++x)
          {
            auto tmp = it->ptr<value_type>(y, x);
            auto p = ptr++;
            for (int c = 0; c < mat.channels(); ++c)
            {
              *p = static_cast<float>(tmp[c]) / (std::numeric_limits<value_type>::max());
              p += offset;
            }
          }
        }

        ptr += offset * (mat.channels() - 1);
      }
    }
  };
}

#endif //!DLIB_INPUT_HPP
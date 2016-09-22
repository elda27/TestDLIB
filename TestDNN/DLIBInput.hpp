#ifndef DLIB_INPUT_HPP
#define DLIB_INPUT_HPP

#include <dlib/dnn/tensor.h>
#include <dlib/dnn/input.h>
#include <dlib/opencv/cv_image.h>

namespace dlib
{
  /*
  template <typename T>
  class input<cv_image<T>>
  {
  public:
    using input_type = cv_image<T> ;
    const static unsigned int sample_expansion_factor = 1;

    template <typename input_iterator>
    void to_tensor(
      input_iterator ibegin,
      input_iterator iend,
      resizable_tensor& data
    ) const
    {
      DLIB_CASSERT(std::distance(ibegin, iend) > 0, "");
      const auto nr = ibegin->nr();
      const auto nc = ibegin->nc();
      // make sure all the input matrices have the same dimensions
      for (auto i = ibegin; i != iend; ++i)
      {
        DLIB_CASSERT(i->nr() == nr && i->nc() == nc,
          "\t input::to_tensor()"
          << "\n\t All matrices given to to_tensor() must have the same dimensions."
          << "\n\t nr: " << nr
          << "\n\t nc: " << nc
          << "\n\t i->nr(): " << i->nr()
          << "\n\t i->nc(): " << i->nc()
        );
      }


      // initialize data to the right size to contain the stuff in the iterator range.
      data.set_size(std::distance(ibegin, iend), pixel_traits<T>::num, nr, nc);

      typedef typename pixel_traits<T>::basic_pixel_type bptype;

      const size_t offset = nr*nc;
      auto ptr = data.host();
      for (auto i = ibegin; i != iend; ++i)
      {
        for (long r = 0; r < nr; ++r)
        {
          for (long c = 0; c < nc; ++c)
          {
            auto temp = pixel_to_vector<float>((*i)(r, c));
            auto p = ptr++;
            for (long j = 0; j < temp.size(); ++j)
            {
              if (is_same_type<bptype, unsigned char>::value)
                *p = temp(j) / 256.0;
              else
                *p = temp(j);
              p += offset;
            }
          }
        }
        ptr += offset*(data.k() - 1);
      }

    }

    friend void serialize(const input& item, std::ostream& out)
    {
      serialize("input<matrix>", out);
    }

    friend void deserialize(input& item, std::istream& in)
    {
      std::string version;
      deserialize(version, in);
      if (version != "input<matrix>")
        throw serialization_error("Unexpected version found while deserializing dlib::input.");
    }

    friend std::ostream& operator<<(std::ostream& out, const input& item)
    {
      out << "input<matrix>";
      return out;
    }

    friend void to_xml(const input& item, std::ostream& out)
    {
      out << "<input/>";
    }
  };
  */
}

#endif //!DLIB_INPUT_HPP
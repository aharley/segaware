#include <vector>

#include "caffe/util/im2parity.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void im2parity_cpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const bool has_ignore_label, const int ignore_label,
    Dtype* data_col) {
  const int height_col = (height + 2 * pad_h - 
			  (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int width_col = (width + 2 * pad_w - 
			 (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int channels_col = channels * kernel_h * kernel_w;



  for (int c_col = 0; c_col < channels_col; ++c_col) {
    int w_offset = c_col % kernel_w;
    int h_offset = (c_col / kernel_w) % kernel_h;
    int c_im = c_col / kernel_h / kernel_w;
    for (int h_col = 0; h_col < height_col; ++h_col) {
      for (int w_col = 0; w_col < width_col; ++w_col) {
        int h_im = h_col * stride_h - pad_h + h_offset * dilation_h;
        int w_im = w_col * stride_w - pad_w + w_offset * dilation_w;
	int center_h_im = h_col * stride_h - pad_h + (kernel_h / 2) * dilation_h;
	int center_w_im = w_col * stride_w - pad_w + (kernel_w / 2) * dilation_w;


	// if you pad by too much, the "center" of the patch may be outside the image. discard info from these.
        if (center_h_im >= 0 && center_h_im < height && 
	    center_w_im >= 0 && center_w_im < width) {
	  int dataCenter = data_im[(c_im * height + center_h_im) * width + center_w_im];
	  int dataHere = 0;
	  // if we're in bounds, replace 0 with actual data
	  if (h_im >= 0 && h_im < height && w_im >= 0 && w_im < width)
	    dataHere = data_im[(c_im * height + h_im) * width + w_im];
	  if (has_ignore_label && (dataHere == ignore_label || dataCenter == ignore_label))
	    data_col[((c_col%(kernel_h*kernel_w)) * height_col + h_col) * width_col + w_col] =
	      ignore_label;
	  else 
	    data_col[((c_col%(kernel_h*kernel_w)) * height_col + h_col) * width_col + w_col] =
	      dataHere==dataCenter; 
	} else {
	  data_col[((c_col%(kernel_h*kernel_w)) * height_col + h_col) * width_col + w_col] =
	    ignore_label;
	}
      }
    }
  }
}

// Explicit instantiation
template void im2parity_cpu<float>(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, 
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const bool has_ignore_label, const int ignore_label,				   
    float* data_col);
template void im2parity_cpu<double>(const double* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const bool has_ignore_label, const int ignore_label,
    double* data_col);

}  // namespace caffe

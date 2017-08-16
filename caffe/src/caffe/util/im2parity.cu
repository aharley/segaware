#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "caffe/common.hpp"
#include "caffe/util/im2parity.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void im2parity_gpu_kernel(const int n, const int channels, const Dtype* data_im,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const bool has_ignore_label, const int ignore_label,
    const int height_col, const int width_col,
    Dtype* data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    const int h_index = index / width_col;
    const int h_col = h_index % height_col;
    const int w_col = index % width_col;
    const int h_offset = h_col * stride_h - pad_h;
    const int w_offset = w_col * stride_w - pad_w;
    for (int c_im = 0; c_im < channels; ++c_im) {

      Dtype* data_col_ptr = data_col;

      data_col_ptr += (h_col) * width_col + w_col;

      const Dtype* data_im_ptr = data_im;
      data_im_ptr += (c_im * height + h_offset) * width + w_offset;
      int center_h_im = h_offset + (kernel_h / 2) * dilation_h;
      int center_w_im = w_offset + (kernel_w / 2) * dilation_w;
      // if you pad by too much, the "center" of the patch may be outside the image. discard info from these.
      if (center_h_im >= 0 && center_h_im < height &&
	  center_w_im >= 0 && center_w_im < width) {
	for (int i = 0; i < kernel_h; ++i) {
	  for (int j = 0; j < kernel_w; ++j) {
	    int h_im = h_offset + i * dilation_h;
	    int w_im = w_offset + j * dilation_w;
	    int dataCenter = data_im_ptr[(kernel_h / 2) * dilation_h * width + 
					 (kernel_w / 2) * dilation_w];
	    int dataHere = 0;
  	    if (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width)
	      dataHere = data_im_ptr[i * dilation_h * width + 
				     j * dilation_w];
	    if (has_ignore_label && 
		(dataHere == ignore_label || 
		 dataCenter == ignore_label)) {
	      *data_col_ptr = ignore_label;
	    } else {
	      *data_col_ptr = dataHere==dataCenter;
	    }
	    data_col_ptr += height_col * width_col; // step into next channel_col
  	  }
  	}
      }
    }
  }
}

template <typename Dtype>
void im2parity_gpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const bool has_ignore_label, const int ignore_label,
    Dtype* data_col) {
  // We are going to launch height_col * width_col kernels, each
  // kernel responsible for turning a multi-channel block into parities.
  // Unlike im2col, we can't kernelize across channels, because all
  // channels contribute to the same spot in data_col. 
  int height_col = (height + 2 * pad_h -
      (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  int width_col = (width + 2 * pad_w -
      (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  int num_kernels = height_col * width_col;

  // NOLINT_NEXT_LINE(whitespace/operators)
  im2parity_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, channels, data_im, height, width, kernel_h, kernel_w, pad_h,
      pad_w, stride_h, stride_w, 
      dilation_h, dilation_w, 
      has_ignore_label, ignore_label, 
      height_col, width_col, 
      data_col);
  CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void im2parity_gpu<float>(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const bool has_ignore_label, const int ignore_label,
    float* data_col);
template void im2parity_gpu<double>(const double* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const bool has_ignore_label, const int ignore_label,
    double* data_col);

}  // namespace caffe

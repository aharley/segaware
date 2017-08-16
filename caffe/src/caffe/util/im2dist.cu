#include <algorithm>
#include <cfloat>

#include "caffe/common.hpp"
#include "caffe/util/im2dist.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void im2dist_gpu_kernel(const int n, const int channels, const Dtype* data_im,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int height_col, const int width_col,
    Dtype* data_col, Dtype* diff_col, int norm, 
    bool remove_center, bool remove_bounds) {
  CUDA_KERNEL_LOOP(index, n) {
    const int h_index = index / width_col;
    const int h_col = h_index % height_col;
    const int w_col = index % width_col;
    const int h_offset = h_col * stride_h - pad_h;
    const int w_offset = w_col * stride_w - pad_w;
    for (int c_im = 0; c_im < channels; ++c_im) {
      int c_col = c_im * kernel_h * kernel_w;
      Dtype* data_col_ptr = data_col;
      Dtype* diff_col_ptr = diff_col;
      data_col_ptr += (h_col) * width_col + w_col;
      diff_col_ptr += (c_col * height_col + h_col) * width_col + w_col;
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

	    // if remove_center is enabled, set the center-center dist as FLT_MAX
	    if (remove_center && h_im == center_h_im && w_im == center_w_im) {
	      *diff_col_ptr = 0;
	      *data_col_ptr = FLT_MAX/100; // /10 just to give room for the grad check
	      // LOG(ERROR) << "setting " << diff_col_index << " in diff and " << data_col_index << " in data to inf";
	      // LOG(ERROR) << "setting " << diff_col_p_index << " in diff and " << data_col_index << " in data to inf";
	    } else {
	      if (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) {
		*diff_col_ptr = data_im_ptr[(kernel_h / 2) * dilation_h * width + 
					    (kernel_w / 2) * dilation_w]; // +center
		*diff_col_ptr -= data_im_ptr[i * dilation_h * width + 
					     j * dilation_w]; // -here
		if (norm==2)
		  *data_col_ptr += (*diff_col_ptr) * (*diff_col_ptr) * 0.5; // sq/2
		else
		  *data_col_ptr += abs(*diff_col_ptr); // abs
	      } else {
		if (remove_bounds)
		  *data_col_ptr = FLT_MAX/100; // if out of bounds, dist is inf
		*diff_col_ptr = 0;
	      }
	    }
	    diff_col_ptr += height_col * width_col; // step into next channel_col
	    data_col_ptr += height_col * width_col; // step into next channel_col
	  }
 	}
      }
    }
  }
}

template <typename Dtype>
void im2dist_gpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    Dtype* data_col, Dtype* diff_col, int norm, 
    bool remove_center, bool remove_bounds) {
  // We are going to launch height_col * width_col kernels, each
  // kernel responsible for turning a multi-channel block into dists.
  // Unlike im2col, we can't kernelize across channels, because all
  // channels contribute to the same spot in data_col. 
  int height_col = (height + 2 * pad_h -
      (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  int width_col = (width + 2 * pad_w -
      (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  int num_kernels = height_col * width_col;

  // Since we compute these incrementally, we need to set them to 0 first.
  caffe_gpu_set(height_col * width_col * kernel_h*kernel_w, Dtype(0), data_col);
  caffe_gpu_set(height_col * width_col * kernel_h*kernel_w*channels, Dtype(0), diff_col);

  // NOLINT_NEXT_LINE(whitespace/operators)
  im2dist_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, channels, data_im, height, width, kernel_h, kernel_w, pad_h,
      pad_w, stride_h, stride_w, dilation_h, dilation_w, 
      height_col, width_col, data_col, diff_col, norm, remove_center, remove_bounds);
  CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void im2dist_gpu<float>(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, 
    float* data_col, float* diff_col, int norm, 
    bool remove_center, bool remove_bounds);
template void im2dist_gpu<double>(const double* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, 
    double* data_col, double* diff_col, int norm,
    bool remove_center, bool remove_bounds);

template <typename Dtype>
__global__ void dist2im_gpu_kernel(const int n, const Dtype* data_col, const Dtype* diff_col,
    const int height, const int width, const int channels,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, 
    const int height_col, const int width_col,
    Dtype* data_im, int norm, 
    bool remove_center, bool remove_bounds) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype val = 0;
    const int w_im = index % width + pad_w;
    const int h_im = (index / width) % height + pad_h;
    const int c_im = index / (width * height);
    int kernel_extent_w = (kernel_w - 1) * dilation_w + 1;
    int kernel_extent_h = (kernel_h - 1) * dilation_h + 1;
    int w_from = w_im - kernel_w/2 * dilation_w;
    int h_from = h_im - kernel_h/2 * dilation_h;
    int w_to = w_im + (kernel_w-1)/2 * dilation_w;
    int h_to = h_im + (kernel_h-1)/2 * dilation_h;
    const int w_col_start = 
      (w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_w + 1;
    const int w_col_end = min(w_im / stride_w + 1, width_col);
    const int h_col_start = 
      (h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) / stride_h + 1;
    const int h_col_end = min(h_im / stride_h + 1, height_col);
    for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
      for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
        int h_k = (h_im - h_col * stride_h);
        int w_k = (w_im - w_col * stride_w);
        if (h_k % dilation_h == 0 && w_k % dilation_w == 0) {
          h_k /= dilation_h;
          w_k /= dilation_w;
          int diff_col_index = (((c_im * kernel_h + h_k) * kernel_w + w_k) *
                                height_col + h_col) * width_col + w_col;
          int data_col_index = (((h_k) * kernel_w + w_k) *
                                height_col + h_col) * width_col + w_col;
  	  if (norm==2)
  	    val -= diff_col[diff_col_index]*data_col[data_col_index];
  	  else if (diff_col[diff_col_index]!=0) 
  	    val -= (1 - 2*Dtype(diff_col[diff_col_index]<0))*data_col[data_col_index];
        }
      }
    }
    // We also need to sum up contributions from the places where this guy was the center pixel. 
    // Conveniently, the values for these contributions are all stored/copied at the depth column
    // centered on this pixel! 
    int channels_col = kernel_h*kernel_w;
    if (h_from % stride_h == 0 && 
	w_from % stride_w == 0 &&
	h_from >= 0 && 
    	w_from >= 0 &&
    	h_to <= height + pad_h * 2 - 1 && 
    	w_to <= width + pad_w * 2 - 1) {
      for (int c_col=0; c_col < channels_col; ++c_col) {
 	int data_col_index = (h_from) * width_col / stride_h + 
 	  (w_from) / stride_w + 
	  c_col * width_col * height_col;
 	int diff_col_index = c_im * height_col * width_col * channels_col + data_col_index;
  	if (norm==2)
  	  val += diff_col[diff_col_index] * data_col[data_col_index];
  	else if (diff_col[diff_col_index]!=0) 
  	  val += (1 - 2*Dtype(diff_col[diff_col_index]<0)) * data_col[data_col_index];
      }
    }
    data_im[index] = val;
  }
}

template <typename Dtype>
void dist2im_gpu(const Dtype* data_col, const Dtype* diff_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w, 
    Dtype* data_im, int norm, 
    bool remove_center, bool remove_bounds) {
  int height_col = (height + 2 * pad_h - 
      (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  int width_col = (width + 2 * pad_w - 
      (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  int num_kernels = channels * height * width;
  caffe_gpu_set(channels * height * width, Dtype(0), data_im);
  dist2im_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                              CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, data_col, diff_col, height, width, channels, kernel_h, kernel_w,
      pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
      height_col, width_col, data_im, norm, remove_center, remove_bounds);
  CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void dist2im_gpu<float>(const float* data_col, const float* diff_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w, 
    const int dilation_h, const int dilation_w, float* data_im, int norm, 
    bool remove_center, bool remove_bounds);
template void dist2im_gpu<double>(const double* data_col, const double* diff_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w, 
    const int dilation_h, const int dilation_w, double* data_im, int norm, 
    bool remove_center, bool remove_bounds);

}  // namespace caffe

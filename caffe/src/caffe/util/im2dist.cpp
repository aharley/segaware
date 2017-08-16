#include <vector>
#include <cfloat>

#include "caffe/util/im2dist.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void im2dist_cpu(const Dtype* data_im, const int channels,
		 const int height, const int width, const int kernel_h, const int kernel_w,
		 const int pad_h, const int pad_w,
		 const int stride_h, const int stride_w,
		 const int dilation_h, const int dilation_w,
		 Dtype* data_col, Dtype* diff_col, int norm, 
		 bool remove_center, bool remove_bounds) {
  const int height_col = (height + 2 * pad_h - 
			  (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int width_col = (width + 2 * pad_w - 
			 (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int channels_col = channels * kernel_h * kernel_w;
  // we compute these incrementally, so we have to init to 0
  caffe_set(height_col * width_col * kernel_h*kernel_w, Dtype(0), data_col);
  caffe_set(height_col * width_col * channels_col, Dtype(0), diff_col);
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
	int diff_col_index = (c_col * height_col + h_col) * width_col + w_col;
	int data_col_index = ((c_col%(kernel_h*kernel_w)) * height_col + h_col) * width_col + w_col;
	// if you pad by too much, the center of the patch may be outside the image. discard info from these.
	if (center_h_im >= 0 && center_h_im < height && 
	    center_w_im >= 0 && center_w_im < width) {
	  // if remove_center is enabled, set the center-center dist as FLT_MAX
	  if (remove_center && h_im == center_h_im && w_im == center_w_im) {
	    diff_col[diff_col_index] = 0; //FLT_MAX/100;
	    data_col[data_col_index] = FLT_MAX/100; // /100 just to give room for the grad check
	  } else {
	    if (h_im >= 0 && h_im < height && w_im >= 0 && w_im < width) {
	      diff_col[diff_col_index] = 
		data_im[(c_im * height + center_h_im) * width + center_w_im]; // +center
	      diff_col[diff_col_index] -= 
		data_im[(c_im * height + h_im) * width + w_im]; // -here
	    if (norm==2)
	      data_col[data_col_index] +=
		diff_col[diff_col_index]*
		diff_col[diff_col_index]*0.5; // sq/2
	    else 
	      data_col[data_col_index] +=
		fabs(Dtype(1.0)*diff_col[diff_col_index]); // abs
	    } else {
	      if (remove_bounds)
		data_col[data_col_index] = FLT_MAX/100; // if out of bounds, dist is inf
	      // diff_col[diff_col_index] = 0;
	    }

	  }
	}
      }
    }
  }
}

// Explicit instantiation
template void im2dist_cpu<float>(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w, 
    const int dilation_h, const int dilation_w, float* data_col, float* diff_col, int norm, 
    bool remove_center, bool remove_bounds);
template void im2dist_cpu<double>(const double* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w, 
    const int dilation_h, const int dilation_w, double* data_col, double* diff_col, int norm,
    bool remove_center, bool remove_bounds);

template <typename Dtype>
void dist2im_cpu(const Dtype* top_diff, const Dtype* diff_col, const int channels,
		 const int height, const int width, const int kernel_h, const int kernel_w,
		 const int pad_h, const int pad_w,
		 const int stride_h, const int stride_w,
		 const int dilation_h, const int dilation_w,
		 Dtype* data_im, int norm, 
		 bool remove_center, bool remove_bounds) {
  // we compute this piecewise, so we have to init to 0
  caffe_set(height * width * channels, Dtype(0), data_im);
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
	int diff_col_index = (c_col * height_col + h_col) * width_col + w_col;
	int top_diff_index = ((c_col%(kernel_h*kernel_w)) * height_col + h_col) * width_col + w_col;
	if (center_h_im >= 0 && center_h_im < height && 
	    center_w_im >= 0 && center_w_im < width ) {
	  // the center of a patch never contributes to its own deriv, since the dist is either 0 or inf;
	  // this happens naturally with the code below, but we can save a bit of time by skipping it.
	  if (h_im == center_h_im && w_im == center_w_im) {
	    // call it a day
	  } else {
	    if (norm==2) {
	      data_im[(c_im * height + center_h_im) * width + center_w_im] +=
		diff_col[diff_col_index] *
		top_diff[top_diff_index]; // +=top*center
	      if (h_im >= 0 && h_im < height && w_im >= 0 && w_im < width)
		data_im[(c_im * height + h_im) * width + w_im] -= 
		  diff_col[diff_col_index] *
		  top_diff[top_diff_index]; // -=top*here
	    } else {
	      data_im[(c_im * height + center_h_im) * width + center_w_im] +=
		(1 - 2*Dtype(diff_col[diff_col_index]<=0)) *
		top_diff[top_diff_index]; // +=top*{-1,1} (for center)
	      if (h_im >= 0 && h_im < height && w_im >= 0 && w_im < width)
		data_im[(c_im * height + h_im) * width + w_im] -= 
		  (1 - 2*Dtype(diff_col[diff_col_index]<=0)) *
		  top_diff[top_diff_index]; // -=top*{-1,1} (for here)
	    }
    	  }
    	}
      }
    }
  }
}


// Explicit instantiation
template void dist2im_cpu<float>(const float* data_col, const float* diff_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, float* data_im, int norm, 
    bool remove_center, bool remove_bounds);
template void dist2im_cpu<double>(const double* data_col, const double* diff_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, double* data_im, int norm,
    bool remove_center, bool remove_bounds);

}  // namespace caffe

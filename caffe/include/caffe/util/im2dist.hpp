#ifndef _CAFFE_UTIL_IM2DIST_HPP_
#define _CAFFE_UTIL_IM2DIST_HPP_

namespace caffe {

template <typename Dtype>
void im2dist_cpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    Dtype* data_col, Dtype* diff_col, int norm, 
    bool remove_center, bool remove_bounds);

template <typename Dtype>
void dist2im_cpu(const Dtype* data_col, const Dtype* diff_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    Dtype* data_im, int norm,
    bool remove_center, bool remove_bounds);

template <typename Dtype>
void im2dist_gpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    Dtype* data_col, Dtype* diff_col, int norm,
    bool remove_center, bool remove_bounds);

template <typename Dtype>
void dist2im_gpu(const Dtype* data_col, const Dtype* diff_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    Dtype* data_im, int norm,
    bool remove_center, bool remove_bounds);

}  // namespace caffe

#endif  // CAFFE_UTIL_IM2DIST_HPP_

#ifndef _CAFFE_UTIL_IM2PARITY_HPP_
#define _CAFFE_UTIL_IM2PARITY_HPP_

namespace caffe {

template <typename Dtype>
void im2parity_cpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w, 
    const int dilation_h, const int dilation_w,
    const bool has_ignore_label, const int ignore_label,
    Dtype* data_col);

template <typename Dtype>
void im2parity_gpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w, 
    const int dilation_h, const int dilation_w,
    const bool has_ignore_label, const int ignore_label,
    Dtype* data_col);

}  // namespace caffe

#endif  // CAFFE_UTIL_IM2PARITY_HPP_

#ifndef CAFFE_NORM_CONV_LAYER_HPP_
#define CAFFE_NORM_CONV_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/im2col.hpp"
#include "caffe/util/im2dist.hpp"

namespace caffe {

/**
 * @brief A helper for image operations that rearranges image regions into
 *        distances from their center. Similar to im2col in blob shapes.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class NormConvLayer : public Layer<Dtype> {
 public:
  explicit NormConvLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "NormConv"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  // virtual inline int equTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  // Helper functions that abstract away the column buffer and gemm arguments.
  void forward_cpu_bias(Dtype* output, const Dtype* bias);
  void backward_cpu_bias(Dtype* bias, const Dtype* input);
  void norm_forward_cpu_gemm(const Dtype* weights, Dtype* output, bool skip_im2col = false);
  void norm_backward_cpu_img_gemm(const Dtype* top_diff, const Dtype* weights, Dtype* output_img);
  void norm_backward_cpu_emb_gemm(const Dtype* top_diff, const Dtype* weights, Dtype* output_emb);
  void norm_weight_cpu_gemm(const Dtype* output, Dtype* weights);
  void backward_cpu_scale(Dtype* scale_diff, const Dtype* weights, const Dtype* emb, const Dtype* top_diff);
//   virtual void norm_backward_cpu_all(const Dtype* top_diff, 
// 				   const Dtype* weights, 
// 				   const Dtype* img, 
// 				   const Dtype* emb, 
// 				   Dtype* weight_diff,
// 				   Dtype* img_diff,
// 				   Dtype* emb_diff,
// 				   Dtype* scale_diff);
  void prep_buffers_cpu(const Dtype* weights, const Dtype* input_img, const Dtype* input_emb);

#ifndef CPU_ONLY
  void prep_buffers_gpu(const Dtype* weights, const Dtype* input_img, const Dtype* input_emb);
  void forward_gpu_bias(Dtype* output, const Dtype* bias);
  void backward_gpu_bias(Dtype* bias, const Dtype* input);
  void norm_forward_gpu_gemm(const Dtype* weights, Dtype* output, bool skip_im2col = false);
  void norm_backward_gpu_img_gemm(const Dtype* top_diff, const Dtype* weights, Dtype* output_img);
  void norm_backward_gpu_emb_gemm(const Dtype* top_diff, const Dtype* weights, Dtype* output_emb);
  void norm_weight_gpu_gemm(const Dtype* output, Dtype* weights);
  void backward_gpu_scale(Dtype* scale_diff, const Dtype* weights, const Dtype* emb, const Dtype* top_diff);
//   virtual void norm_backward_gpu_all(const Dtype* top_diff, 
// 				   const Dtype* weights, 
// 				   const Dtype* img, 
// 				   const Dtype* emb, 
// 				   Dtype* weight_diff,
// 				   Dtype* img_diff,
// 				   Dtype* emb_diff,
// 				   Dtype* scale_diff);
#endif

  /// @brief The spatial dimensions of the input.
  inline int input_shape(int i) {
    return (*bottom_shape_)[channel_axis_ + i];
  }
  virtual inline bool reverse_dimensions() { return false; }
  virtual void compute_output_shape();

  /// @brief The spatial dimensions of a filter kernel.
  Blob<int> kernel_shape_;
  /// @brief The spatial dimensions of the stride.
  Blob<int> stride_;
  /// @brief The spatial dimensions of the padding.
  Blob<int> pad_;
  /// @brief The spatial dimensions of the dilation.
  Blob<int> dilation_;
  /// @brief The spatial dimensions of the convolution input image.
  Blob<int> conv_input_shape_;
  /// @brief The spatial dimensions of the convolution input embedding.
  Blob<int> emb_input_shape_;
  /// @brief The spatial dimensions of the col_buffer.
  vector<int> col_buffer_shape_;
  /// @brief The spatial dimensions of the emb_col_buffer, for normalized conv.
  vector<int> emb_col_buffer_shape_;
  /// @brief The spatial dimensions of the diff_col_buffer, for normalized conv.
  vector<int> diff_col_buffer_shape_;
  /// @brief The spatial dimensions of the output.
  vector<int> output_shape_;
  /// @brief The spatial dimensions of the sum_buffer, for normalized conv.
  vector<int> sum_buffer_shape_;
  const vector<int>* bottom_shape_;

  int num_spatial_axes_;
  int bottom_dim_;
  int emb_bottom_dim_;
  int top_dim_;

  int channel_axis_;
  int num_;
  int channels_;
  int emb_channels_;
  int group_;
  int in_spatial_dim_;
  int out_spatial_dim_;
  int weight_offset_;
  int num_output_;
  int norm_;
  bool remove_center_;
  bool remove_bounds_;
  int scale_ind_;
  bool bias_term_;
  bool scale_term_;
  bool is_1x1_;
  bool force_nd_im2col_;
  bool bottom_is_im2col_;
  bool abs_scale_;

private:
  // wrap im2col/col2im so we don't have to remember the (long) argument lists
  inline void conv_im2col_cpu(const Dtype* data, Dtype* col_buff) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      im2col_cpu(data, conv_in_channels_,
          conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1],
          dilation_.cpu_data()[0], dilation_.cpu_data()[1], col_buff);
    } else {
      im2col_nd_cpu(data, num_spatial_axes_, conv_input_shape_.cpu_data(),
          col_buffer_shape_.data(), kernel_shape_.cpu_data(),
          pad_.cpu_data(), stride_.cpu_data(), dilation_.cpu_data(), col_buff);
    }
  }
  inline void conv_col2im_cpu(const Dtype* col_buff, Dtype* data) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      col2im_cpu(col_buff, conv_in_channels_,
          conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1],
          dilation_.cpu_data()[0], dilation_.cpu_data()[1], data);
    } else {
      col2im_nd_cpu(col_buff, num_spatial_axes_, conv_input_shape_.cpu_data(),
          col_buffer_shape_.data(), kernel_shape_.cpu_data(),
          pad_.cpu_data(), stride_.cpu_data(), dilation_.cpu_data(), data);
    }
  }
#ifndef CPU_ONLY
  inline void conv_im2col_gpu(const Dtype* data, Dtype* col_buff) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      im2col_gpu(data, conv_in_channels_,
          conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1],
          dilation_.cpu_data()[0], dilation_.cpu_data()[1], col_buff);
    } else {
      im2col_nd_gpu(data, num_spatial_axes_, num_kernels_im2col_,
          conv_input_shape_.gpu_data(), col_buffer_.gpu_shape(),
          kernel_shape_.gpu_data(), pad_.gpu_data(),
          stride_.gpu_data(), dilation_.gpu_data(), col_buff);
    }
  }
  inline void conv_col2im_gpu(const Dtype* col_buff, Dtype* data) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      col2im_gpu(col_buff, conv_in_channels_,
          conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1],
          dilation_.cpu_data()[0], dilation_.cpu_data()[1], data);
    } else {
      col2im_nd_gpu(col_buff, num_spatial_axes_, num_kernels_col2im_,
          conv_input_shape_.gpu_data(), col_buffer_.gpu_shape(),
          kernel_shape_.gpu_data(), pad_.gpu_data(), stride_.gpu_data(),
          dilation_.gpu_data(), data);
    }
  }
#endif
  // wrap im2dist/dist2im so we don't have to remember the (long) argument lists
  inline void conv_im2dist_cpu(const Dtype* data, Dtype* col_buff, Dtype* diff_col_buff) {
    im2dist_cpu(data, emb_channels_,
		emb_input_shape_.cpu_data()[1], emb_input_shape_.cpu_data()[2],
		kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
		pad_.cpu_data()[0], pad_.cpu_data()[1],
		stride_.cpu_data()[0], stride_.cpu_data()[1],
		dilation_.cpu_data()[0], dilation_.cpu_data()[1], 
		col_buff, diff_col_buff, norm_, 
		remove_center_, remove_bounds_);
  }
  inline void conv_dist2im_cpu(const Dtype* col_buff, const Dtype* diff_col_buff, Dtype* data) {
    dist2im_cpu(col_buff, diff_col_buff, emb_channels_,
		emb_input_shape_.cpu_data()[1], emb_input_shape_.cpu_data()[2],
		kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
		pad_.cpu_data()[0], pad_.cpu_data()[1],
		stride_.cpu_data()[0], stride_.cpu_data()[1],
		dilation_.cpu_data()[0], dilation_.cpu_data()[1], data, norm_, 
		remove_center_, remove_bounds_);
  }
#ifndef CPU_ONLY
  inline void conv_im2dist_gpu(const Dtype* data, Dtype* col_buff, Dtype* diff_col_buff) {
    im2dist_gpu(data, emb_channels_,
		emb_input_shape_.cpu_data()[1], emb_input_shape_.cpu_data()[2],
		kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
		pad_.cpu_data()[0], pad_.cpu_data()[1],
		stride_.cpu_data()[0], stride_.cpu_data()[1],
		dilation_.cpu_data()[0], dilation_.cpu_data()[1], 
		col_buff, diff_col_buff, norm_, 
		remove_center_, remove_bounds_);
  }
  inline void conv_dist2im_gpu(const Dtype* col_buff, const Dtype* diff_col_buff, Dtype* data) {
    dist2im_gpu(col_buff, diff_col_buff, emb_channels_,
		emb_input_shape_.cpu_data()[1], emb_input_shape_.cpu_data()[2],
		kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
		pad_.cpu_data()[0], pad_.cpu_data()[1],
		stride_.cpu_data()[0], stride_.cpu_data()[1],
		dilation_.cpu_data()[0], dilation_.cpu_data()[1], data, norm_, 
		remove_center_, remove_bounds_);
  }
#endif
  int num_kernels_im2col_;
  int num_kernels_col2im_;
  int conv_out_channels_;
  int conv_in_channels_;
  int conv_out_spatial_dim_;
  int kernel_dim_;
  int col_offset_;
  int output_offset_;

  Blob<Dtype> col_buffer_; 
  Blob<Dtype> emb_col_buffer_; // for the result of emb->im2dist in normalized conv
  Blob<Dtype> soft_col_buffer_; // for the result of softmax(embdist) in normalized conv
  Blob<Dtype> saved_emb_col_buffer_; // for the result of emb->im2dist in normalized conv
  Blob<Dtype> noexp_emb_col_buffer_; // for the result of emb->im2dist in normalized conv
  Blob<Dtype> res_col_buffer_; // for the product of img->im2col*emb->im2dist in normalized_conv
  Blob<Dtype> saved_res_col_buffer_; // for the product of img->im2col*emb->im2dist in normalized_conv
  Blob<Dtype> diff_col_buffer_; // for the actual differences produced in emb->im2dist
  Blob<Dtype> sum_buffer_; // for the sum of each filter (after im2dist and exp)
  Blob<Dtype> sum_multiplier_;
};

}  // namespace caffe

#endif  // CAFFE_NORM_CONV_LAYER_HPP_

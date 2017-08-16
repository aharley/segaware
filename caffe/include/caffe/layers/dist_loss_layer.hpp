#ifndef CAFFE_DIST_LOSS_LAYER_HPP_
#define CAFFE_DIST_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
 * @brief Computes a distance loss, using im2dist and im2parity outputs.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */

template <typename Dtype>
class DistLossLayer : public LossLayer<Dtype> {
 public:
  explicit DistLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "DistLoss"; }
  /**
   * @param param provides DistParameter dist_loss_param, with options:
   *  - ignore_label (optional)
   *    Specify a label value that should be ignored when computing the loss.
   *  - num_output (required): the number of different labels to watch for.
   *  - kernel_size
   *  - stride
   *  - alpha
   *  - beta
   */
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Dtype alpha_;
  Dtype beta_;

  /// @brief The spatial dimensions of a filter kernel.
  Blob<int> kernel_shape_;
  /// @brief The spatial dimensions of the stride.
  Blob<int> stride_;
  /// @brief The spatial dimensions of the padding.
  Blob<int> pad_;
  /// @brief The spatial dimensions of the dilation.
  Blob<int> dilation_;
  /// @brief Differences, saved for the backward pass. 
  Blob<Dtype> diff_;

  int num_spatial_axes_;
  int bottom_dim_;
  int top_dim_;
  int diff_dim_;

  int channel_axis_;
  int num_;
  int channels_;

  // Whether to ignore instances with a certain label.
  bool has_ignore_label_;
  // The label indicating that an instance should be ignored.
  int ignore_label_;
  // Whether to normalize the loss by the total number of values present
  // (otherwise just by the batch size).
  bool normalize_; // hmm... i'm not using this yet.
};

}  // namespace caffe

#endif  // CAFFE_DIST_LOSS_LAYER_HPP_

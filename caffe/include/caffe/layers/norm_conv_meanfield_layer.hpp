#ifndef CAFFE_NORM_CONV_MEANFIELD_LAYER_HPP_
#define CAFFE_NORM_CONV_MEANFIELD_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

//#include "caffe/util/modified_permutohedral.hpp"
#include "caffe/layers/norm_conv_meanfield_iteration.hpp"
#include <boost/shared_array.hpp>

#include "caffe/layers/im2dist_layer.hpp"
#include "caffe/layers/im2col_layer.hpp"
#include "caffe/layers/channel_reduce_layer.hpp"
#include "caffe/layers/eltwise_layer.hpp"

namespace caffe {

template <typename Dtype>
class NormConvMeanfieldLayer : public Layer<Dtype> {

 public:
  explicit NormConvMeanfieldLayer(const LayerParameter& param) : Layer<Dtype>(param) {}

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const {
    return "NormConvMeanfield";
  }
  virtual inline int ExactNumBottomBlobs() const { return 4; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  // virtual void compute_spatial_kernel(float* const output_kernel);
  // virtual void compute_bilateral_kernel(const Blob<Dtype>* const rgb_blob, const int n, float* const output_kernel);

  int count_;
  int num_;
  int channels_;
  int height_;
  int width_;
  int num_pixels_;

  Blob<int> kernel_shape_;
  Blob<int> stride_;
  Blob<int> pad_;
  Blob<int> dilation_;
  Blob<Dtype> scale_;
  Blob<unsigned int> remove_center_; // bool don't work
  Blob<unsigned int> remove_bounds_; // bool don't work

  // vector<Blob<Dtype>*> scale_;
  // Dtype 
  // vector<shared_ptr<Blob<Dtype> > > scale_;
  // bool scale_term_;
  // vector<Dtype> scales_;
  // Blob<Dtype> scales_;
  // Blob<bool> remove_centers_;

  // Dtype theta_alpha_;
  // Dtype theta_beta_;
  // Dtype theta_gamma_;
  int num_iterations_;
  // int iter_multiplier_;

  Dtype spatial_scale_;
  Dtype bilateral_scale_;

  boost::shared_array<Dtype> norm_feed_;
  Blob<Dtype> spatial_norm_;
  Blob<Dtype> bilateral_norms_;

  // vector<Blob<Dtype>*> im2dist_layer_bottom_vec_;
  // vector<Blob<Dtype>*> im2dist_layer_top_vec_;
  // vector<shared_ptr<Blob<Dtype> > > im2dist_layer_out_blobs_;

  // vector<Blob<Dtype>*> im2col_layer_bottom_vec_;
  // vector<Blob<Dtype>*> im2col_layer_top_vec_;
  // vector<shared_ptr<Blob<Dtype> > > im2col_layer_out_blobs_;

  vector<Blob<Dtype>*> unary_split_layer_bottom_vec_;
  vector<Blob<Dtype>*> unary_split_layer_top_vec_;
  vector<shared_ptr<Blob<Dtype> > > unary_split_layer_out_blobs_;
  shared_ptr<SplitLayer<Dtype> > unary_split_layer_;

  vector<Blob<Dtype>*> rgbxy_split_layer_bottom_vec_;
  vector<Blob<Dtype>*> rgbxy_split_layer_top_vec_;
  vector<shared_ptr<Blob<Dtype> > > rgbxy_split_layer_out_blobs_;
  shared_ptr<SplitLayer<Dtype> > rgbxy_split_layer_;

  vector<Blob<Dtype>*> xy_split_layer_bottom_vec_;
  vector<Blob<Dtype>*> xy_split_layer_top_vec_;
  vector<shared_ptr<Blob<Dtype> > > xy_split_layer_out_blobs_;
  shared_ptr<SplitLayer<Dtype> > xy_split_layer_;

  vector<shared_ptr<Blob<Dtype> > > iteration_output_blobs_;
  vector<shared_ptr<NormConvMeanfieldIteration<Dtype> > > meanfield_iterations_;

  // shared_ptr<Im2distLayer<Dtype> > im2dist_layer_;
  // shared_ptr<Im2colLayer<Dtype> > im2col_layer_;

  //shared_ptr<ModifiedPermutohedral> spatial_lattice_;
  boost::shared_array<float> bilateral_kernel_buffer_;
  //vector<shared_ptr<ModifiedPermutohedral> > bilateral_lattices_;
};

}  // namespace caffe

#endif  // CAFFE_NORM_CONV_MEANFIELD_LAYER_HPP_

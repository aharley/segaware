#ifndef CAFFE_NORM_CONV_MEANFIELD_ITERATION_HPP_
#define CAFFE_NORM_CONV_MEANFIELD_ITERATION_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

//#include "caffe/util/modified_permutohedral.hpp"
#include <boost/shared_array.hpp>
#include "caffe/layers/eltwise_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"
#include "caffe/layers/split_layer.hpp"
// #include "caffe/layers/im2dist_layer.hpp"
// #include "caffe/layers/tile_layer.hpp"
// #include "caffe/layers/im2col_layer.hpp"
// #include "caffe/layers/channel_reduce_layer.hpp"
#include "caffe/layers/norm_conv_layer.hpp"

namespace caffe {

template <typename Dtype>
class NormConvMeanfieldIteration {

 public:
  /**
   * Must be invoked only once after the construction of the layer.
   */
  void OneTimeSetUp(
      Blob<Dtype>* const unary_terms,
      Blob<Dtype>* const softmax_input,
      Blob<Dtype>* const rgbxy,
      Blob<Dtype>* const positions,
      Blob<int>* const kernel_shape,
      Blob<int>* const stride,
      Blob<int>* const pad,
      Blob<int>* const dilation,
      Blob<Dtype>* const scales,
      Blob<unsigned int>* const remove_center,
      Blob<unsigned int>* const remove_bounds,
      Blob<Dtype>* const output_blob);
      // const shared_ptr<ModifiedPermutohedral> spatial_lattice,
      // const Blob<Dtype>* const spatial_norm);
      // const int iter_multiplier,
      // Blob<bool>* const remove_centers,
  /**
   * Must be invoked before invoking {@link Forward_cpu()}
   */
  virtual void PrePass(
      const vector<shared_ptr<Blob<Dtype> > >&  parameters_to_copy_from);
      // const vector<shared_ptr<ModifiedPermutohedral> >* const bilateral_lattices,
      // const Blob<Dtype>* const bilateral_norms);

  virtual void Forward_cpu();
  virtual void Backward_cpu();
  // virtual void Forward_gpu();
  // virtual void Backward_gpu();
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  // A quick hack. This should be properly encapsulated.
  vector<shared_ptr<Blob<Dtype> > >& blobs() {
    return blobs_;
  }

 protected:
  vector<shared_ptr<Blob<Dtype> > > blobs_;

  int count_;
  int num_;
  int channels_;
  int height_;
  int width_;
  int num_pixels_;

  Blob<Dtype> spatial_out_blob_;
  // Blob<Dtype> rgbxy_out_blob_;
  Blob<Dtype> bilateral_out_blob_;
  Blob<Dtype> pairwise_;
  Blob<Dtype> softmax_input_;
  Blob<Dtype> positions_;
  Blob<Dtype> position_distances_;
  Blob<Dtype> exp_position_distances_;
  Blob<Dtype> tiled_position_distances_;
  Blob<Dtype> prob_distances_;
  Blob<Dtype> hit_distances_;
  Blob<Dtype> reduced_;
  Blob<Dtype> prob_;
  Blob<Dtype> message_passing_;

  vector<Blob<Dtype>*> softmax_bottom_vec_;
  vector<Blob<Dtype>*> softmax_top_vec_;
  vector<Blob<Dtype>*> sum_bottom_vec_;
  vector<Blob<Dtype>*> sum_top_vec_;

  // vector<Blob<Dtype>*> im2dist_bottom_vec_;
  // vector<Blob<Dtype>*> im2dist_top_vec_;
  // vector<Blob<Dtype>*> exp_bottom_vec_;
  // vector<Blob<Dtype>*> exp_top_vec_;
  // vector<Blob<Dtype>*> tile_bottom_vec_;
  // vector<Blob<Dtype>*> tile_top_vec_;
  // vector<Blob<Dtype>*> im2col_bottom_vec_;
  // vector<Blob<Dtype>*> im2col_top_vec_;
  vector<Blob<Dtype>*> eltwise_bottom_vec_;
  vector<Blob<Dtype>*> eltwise_top_vec_;
  // vector<Blob<Dtype>*> channel_reduce_bottom_vec_;
  // vector<Blob<Dtype>*> channel_reduce_top_vec_;

  vector<Blob<Dtype>*> split_layer_bottom_vec_;
  vector<Blob<Dtype>*> split_layer_top_vec_;
  vector<shared_ptr<Blob<Dtype> > > split_layer_out_blobs_;

  vector<Blob<Dtype>*> xy_conv_bottom_vec_;
  vector<Blob<Dtype>*> xy_conv_top_vec_;
  shared_ptr<NormConvLayer<Dtype> > xy_conv_layer_;

  vector<Blob<Dtype>*> rgbxy_conv_bottom_vec_;
  vector<Blob<Dtype>*> rgbxy_conv_top_vec_;
  shared_ptr<NormConvLayer<Dtype> > rgbxy_conv_layer_;



  // shared_ptr<Im2distLayer<Dtype> > im2dist_layer_;
  // shared_ptr<ExpLayer<Dtype> > exp_layer_;
  // shared_ptr<TileLayer<Dtype> > tile_layer_;
  // shared_ptr<Im2colLayer<Dtype> > im2col_layer_;
  // shared_ptr<EltwiseLayer<Dtype> > eltwise_layer_;
  // shared_ptr<ChannelReduceLayer<Dtype> > channel_reduce_layer_;

  shared_ptr<SplitLayer<Dtype> > split_layer_;
  shared_ptr<SoftmaxLayer<Dtype> > softmax_layer_;
  shared_ptr<EltwiseLayer<Dtype> > sum_layer_;

  // shared_ptr<ModifiedPermutohedral> spatial_lattice_;
  // const vector<shared_ptr<ModifiedPermutohedral> >* bilateral_lattices_;

  const Blob<Dtype>* spatial_norm_;
  const Blob<Dtype>* bilateral_norms_;

};

}  // namespace caffe

#endif  // CAFFE_NORM_CONV_MEANFIELD_ITERATION_HPP_

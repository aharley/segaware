/*
 *  \brief     A helper class for {@link MultiStageNormConvMeanfieldLayer} class, which is the Caffe layer that implements the
 *             CRF-RNN described in the paper: Conditional Random Fields as Recurrent Neural Networks. IEEE ICCV 2015.
 *
 *             This class itself is not a proper Caffe layer although it behaves like one to some degree.
 *
 *  \authors   Sadeep Jayasumana, Bernardino Romera-Paredes, Shuai Zheng, Zhizhong Su.
 *  \version   1.0
 *  \date      2015
 *  \copyright Torr Vision Group, University of Oxford.
 *  \details   If you use this code, please consider citing the paper:
 *             Shuai Zheng, Sadeep Jayasumana, Bernardino Romera-Paredes, Vibhav Vineet, Zhizhong Su, Dalong Du,
 *             Chang Huang, Philip H. S. Torr. Conditional Random Fields as Recurrent Neural Networks. IEEE ICCV 2015.
 *
 *             For more information about CRF-RNN, please visit the project website http://crfasrnn.torr.vision.
 */
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/norm_conv_meanfield_iteration.hpp"

namespace caffe {

/**
 * Forward pass during the inference.
 */
template <typename Dtype>
void NormConvMeanfieldIteration<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
// void NormConvMeanfieldIteration<Dtype>::Forward_gpu() {
  //------------------------------- Softmax normalization--------------------
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  //-----------------------------------Message passing-----------------------
  caffe_gpu_set(count_, Dtype(0.), message_passing_.mutable_gpu_data());

  split_layer_->Forward(split_layer_bottom_vec_, split_layer_top_vec_);

  // spatial stuff
  xy_conv_layer_->Forward(xy_conv_bottom_vec_, xy_conv_top_vec_);
  for (int n = 0; n < num_; ++n) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_, num_pixels_, channels_, (Dtype) 1.,
        this->blobs_[0]->gpu_data(), spatial_out_blob_.gpu_data() + spatial_out_blob_.offset(n), (Dtype) 0.,
        message_passing_.mutable_gpu_data() + message_passing_.offset(n));
  }
  // bilateral stuff
  rgbxy_conv_layer_->Forward(rgbxy_conv_bottom_vec_, rgbxy_conv_top_vec_);
  for (int n = 0; n < num_; ++n) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_, num_pixels_, channels_, (Dtype) 1.,
        this->blobs_[1]->gpu_data(), bilateral_out_blob_.gpu_data() + bilateral_out_blob_.offset(n), (Dtype) 1.,
        message_passing_.mutable_gpu_data() + message_passing_.offset(n));
  }
  //--------------------------- Compatibility multiplication ----------------
  // Result from message passing needs to be multiplied with compatibility values.
  for (int n = 0; n < num_; ++n) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_, num_pixels_,
        channels_, (Dtype) 1., this->blobs_[2]->gpu_data(),
        message_passing_.gpu_data() + message_passing_.offset(n), (Dtype) 0.,
        pairwise_.mutable_gpu_data() + pairwise_.offset(n));
  }

  //------------------------- Adding unaries, normalization is left to the next iteration --------------
  // Add unary
  sum_layer_->Forward(sum_bottom_vec_, sum_top_vec_);
}

template<typename Dtype>
void NormConvMeanfieldIteration<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
// void NormConvMeanfieldIteration<Dtype>::Backward_gpu() {
  //---------------------------- Add unary gradient --------------------------
  vector<bool> eltwise_propagate_down(2, true);
  sum_layer_->Backward(sum_top_vec_, eltwise_propagate_down, sum_bottom_vec_);

  //---------------------------- Update compatibility diffs ------------------
  caffe_gpu_set(this->blobs_[2]->count(), Dtype(0.), this->blobs_[2]->mutable_gpu_diff());

  for (int n = 0; n < num_; ++n) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, channels_, channels_,
                          num_pixels_, (Dtype) 1., pairwise_.gpu_diff() + pairwise_.offset(n),
                          message_passing_.gpu_data() + message_passing_.offset(n), (Dtype) 1.,
                          this->blobs_[2]->mutable_gpu_diff());
  }

  //-------------------------- Gradient after compatibility transform--- -----
  for (int n = 0; n < num_; ++n) {
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, channels_, num_pixels_,
                          channels_, (Dtype) 1., this->blobs_[2]->gpu_data(),
                          pairwise_.gpu_diff() + pairwise_.offset(n), (Dtype) 0.,
                          message_passing_.mutable_gpu_diff() + message_passing_.offset(n));
  }

  // ------------------------- Gradient w.r.t. kernels weights ------------
  caffe_gpu_set(this->blobs_[0]->count(), Dtype(0.), this->blobs_[0]->mutable_gpu_diff());
  caffe_gpu_set(this->blobs_[1]->count(), Dtype(0.), this->blobs_[1]->mutable_gpu_diff());

  for (int n = 0; n < num_; ++n) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, channels_, channels_,
                          num_pixels_, (Dtype) 1., message_passing_.gpu_diff() + message_passing_.offset(n),
                          spatial_out_blob_.gpu_data() + spatial_out_blob_.offset(n), (Dtype) 1.,
                          this->blobs_[0]->mutable_gpu_diff());
  }

  for (int n = 0; n < num_; ++n) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, channels_, channels_,
                          num_pixels_, (Dtype) 1., message_passing_.gpu_diff() + message_passing_.offset(n),
                          bilateral_out_blob_.gpu_data() + bilateral_out_blob_.offset(n), (Dtype) 1.,
                          this->blobs_[1]->mutable_gpu_diff());
  }

  for (int n = 0; n < num_; ++n) {
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, channels_, num_pixels_, channels_, (Dtype) 1.,
                          this->blobs_[0]->gpu_data(), message_passing_.gpu_diff() + message_passing_.offset(n),
                          (Dtype) 0.,
                          spatial_out_blob_.mutable_gpu_diff() + spatial_out_blob_.offset(n));
  }

  for (int n = 0; n < num_; ++n) {
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, channels_, num_pixels_, channels_, (Dtype) 1.,
                          this->blobs_[1]->gpu_data(), message_passing_.gpu_diff() + message_passing_.offset(n),
                          (Dtype) 0.,
                          bilateral_out_blob_.mutable_gpu_diff() + bilateral_out_blob_.offset(n));
  }


  //--------------------------- Gradient for message passing ---------------
  vector<bool> prop_down(2, true);
  xy_conv_layer_->Backward(xy_conv_top_vec_, prop_down, xy_conv_bottom_vec_);
  rgbxy_conv_layer_->Backward(rgbxy_conv_top_vec_, prop_down, rgbxy_conv_bottom_vec_);

  vector<bool> split_down(1, true);
  split_layer_->Backward(split_layer_top_vec_, split_down, split_layer_bottom_vec_);

  //--------------------------------------------------------------------------------
  vector<bool> soft_down(2, true);
  softmax_layer_->Backward(softmax_top_vec_, soft_down, softmax_bottom_vec_);
}

INSTANTIATE_LAYER_GPU_FUNCS(NormConvMeanfieldIteration);

}  // namespace caffe

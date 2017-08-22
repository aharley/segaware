/*!
 *  \brief     The Caffe layer that implements the CRF-RNN described in the paper:
 *             Conditional Random Fields as Recurrent Neural Networks. IEEE ICCV 2015.
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
#include "caffe/util/im2col.hpp"
#include "caffe/util/tvg_util.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/nosp_conv_meanfield_layer.hpp"

#include <cmath>

namespace caffe {

/**
 * Performs filter-based mean field inference given the image and unaries.
 *
 * bottom[0] - Unary terms
 * bottom[1] - Softmax input, or if i > 1, the output from the previous iteration.
 * bottom[2] - RGBXY images -- for the lattices, for n=1, it doesn't matter if it's rgb or rgbxy
 *
 * top[0] - Output of the mean field inference (not normalized).
 */
template <typename Dtype>
void NospConvMeanfieldLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  // LOG(ERROR) << "pos_count = " << pos_count;
  // // const Dtype* pos_data = bottom[3]->gpu_data();
  // for (int i = 0; i < 10; i++) {
  //   LOG(ERROR) << "pos_data[" << i << "] = " << pos_data[i];
  //   // LOG(ERROR) << "xy data[" << i << "] = " << xy_[i];
  // }
  
  // for (int i = bottom[3]->count()-10; i < bottom[3]->count(); i++) {
  //   LOG(ERROR) << "pos_data[" << i << "] = " << pos_data[i];
  //   // LOG(ERROR) << "xy data[" << i << "] = " << xy_[i];
  // }
  // LOG(ERROR) << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!";

  unary_split_layer_bottom_vec_[0] = bottom[0];
  unary_split_layer_->Forward(unary_split_layer_bottom_vec_, unary_split_layer_top_vec_);

  rgbxy_split_layer_bottom_vec_[0] = bottom[2];
  rgbxy_split_layer_->Forward(rgbxy_split_layer_bottom_vec_, rgbxy_split_layer_top_vec_);

  // // Initialize the bilateral lattices.
  // bilateral_lattices_.resize(num_);
  // for (int n = 0; n < num_; ++n) {

  //   compute_bilateral_kernel(bottom[2], n, bilateral_kernel_buffer_.get());
  //   bilateral_lattices_[n].reset(new ModifiedPermutohedral());
  //   bilateral_lattices_[n]->init(bilateral_kernel_buffer_.get(), 5, num_pixels_);

  //   // Calculate bilateral filter normalization factors.
  //   Dtype* norm_output_data = bilateral_norms_.mutable_gpu_data() + bilateral_norms_.offset(n);
  //   bilateral_lattices_[n]->compute(norm_output_data, norm_feed_.get(), 1);
  //   for (int i = 0; i < num_pixels_; ++i) {
  //     norm_output_data[i] = 1.f / (norm_output_data[i] + 1e-20f);
  //   }
  // }

  for (int i = 0; i < num_iterations_; ++i) {
    // LOG(ERROR) << "--------- iteration " << i << " ------------------";
    // meanfield_iterations_[i]->PrePass(this->blobs_, &bilateral_lattices_, &bilateral_norms_);
    meanfield_iterations_[i]->PrePass(this->blobs_);
    meanfield_iterations_[i]->Forward_gpu(bottom,top);
  }
}

/**
 * Backprop through filter-based mean field inference.
 */
template<typename Dtype>
void NospConvMeanfieldLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  for (int i = (num_iterations_ - 1); i >= 0; --i) {
    meanfield_iterations_[i]->Backward_gpu(top,propagate_down,bottom);
  }
  vector<bool> split_layer_propagate_down(1, true);
  unary_split_layer_->Backward(unary_split_layer_top_vec_, split_layer_propagate_down, unary_split_layer_bottom_vec_);
  rgbxy_split_layer_->Backward(rgbxy_split_layer_top_vec_, split_layer_propagate_down, rgbxy_split_layer_bottom_vec_);
  // Accumulate diffs from mean field iters for the parameter blobs.
  for (int blob_id = 0; blob_id < this->blobs_.size(); ++blob_id) {
    Blob<Dtype>* cur_blob = this->blobs_[blob_id].get();
    if (this->param_propagate_down_[blob_id]) {
      caffe_gpu_set(cur_blob->count(), Dtype(0), cur_blob->mutable_gpu_diff());
      for (int i = 0; i < num_iterations_; ++i) {
        const Dtype* diffs_to_add = meanfield_iterations_[i]->blobs()[blob_id]->gpu_diff();
        caffe_gpu_axpy(cur_blob->count(), Dtype(1.), diffs_to_add, cur_blob->mutable_gpu_diff());
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(NospConvMeanfieldLayer);
}  // namespace caffe

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
 * To be invoked once only immediately after construction.
 */
template <typename Dtype>
void NormConvMeanfieldIteration<Dtype>::OneTimeSetUp(
    Blob<Dtype>* const unary_terms,
    Blob<Dtype>* const softmax_input,
    Blob<Dtype>* const rgbxy,
    Blob<Dtype>* const xy,
    Blob<int>* const kernel_shape,
    Blob<int>* const stride,
    Blob<int>* const pad,
    Blob<int>* const dilation,
    Blob<Dtype>* const scale,
    Blob<unsigned int>* const remove_center,
    Blob<unsigned int>* const remove_bounds,
    Blob<Dtype>* const output_blob) {
    // const shared_ptr<ModifiedPermutohedral> spatial_lattice,
    // const Blob<Dtype>* const spatial_norm) {
  // LOG(ERROR) << "-------------------------";
  // LOG(ERROR) << "ok, doing one-time setup";
  // LOG(ERROR) << "-------------------------";

  // spatial_lattice_ = spatial_lattice;
  // spatial_norm_ = spatial_norm;

  count_ = unary_terms->count();
  num_ = unary_terms->num();
  channels_ = unary_terms->channels();
  height_ = unary_terms->height();
  width_ = unary_terms->width();
  num_pixels_ = height_ * width_;

  if (this->blobs_.size() > 0) {
    LOG(INFO) << "NormConvMeanfield iteration skipping parameter initialization.";
  } else {
    blobs_.resize(3);
    blobs_[0].reset(new Blob<Dtype>(1, 1, channels_, channels_)); // spatial kernel weight
    blobs_[1].reset(new Blob<Dtype>(1, 1, channels_, channels_)); // bilateral kernel weight
    blobs_[2].reset(new Blob<Dtype>(1, 1, channels_, channels_)); // compatibility transform matrix
  }

  pairwise_.Reshape(num_, channels_, height_, width_);
  spatial_out_blob_.Reshape(num_, channels_, height_, width_);
  bilateral_out_blob_.Reshape(num_, channels_, height_, width_);
  message_passing_.Reshape(num_, channels_, height_, width_);

  // Softmax layer configuration
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(softmax_input);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  LayerParameter softmax_param;
  softmax_layer_.reset(new SoftmaxLayer<Dtype>(softmax_param));
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  // Sum layer configuration
  sum_bottom_vec_.clear();
  sum_bottom_vec_.push_back(unary_terms);
  sum_bottom_vec_.push_back(&pairwise_);
  sum_top_vec_.clear();
  sum_top_vec_.push_back(output_blob);
  LayerParameter sum_param;
  sum_param.mutable_eltwise_param()->add_coeff(Dtype(1.));
  sum_param.mutable_eltwise_param()->add_coeff(Dtype(-1.));
  sum_param.mutable_eltwise_param()->set_operation(EltwiseParameter_EltwiseOp_SUM);
  sum_layer_.reset(new EltwiseLayer<Dtype>(sum_param));
  sum_layer_->SetUp(sum_bottom_vec_, sum_top_vec_);

  // both the spatial (xy) and bilateral (rgbxy) need &prob as bottom. we need to split. 
  split_layer_bottom_vec_.clear();
  split_layer_bottom_vec_.push_back(&prob_);
  split_layer_top_vec_.clear();
  split_layer_out_blobs_.resize(2);
  split_layer_out_blobs_[0].reset(new Blob<Dtype>());
  split_layer_out_blobs_[1].reset(new Blob<Dtype>());
  split_layer_top_vec_.push_back(split_layer_out_blobs_[0].get());
  split_layer_top_vec_.push_back(split_layer_out_blobs_[1].get());
  LayerParameter split_layer_param;
  split_layer_.reset(new SplitLayer<Dtype>(split_layer_param));
  split_layer_->SetUp(split_layer_bottom_vec_, split_layer_top_vec_);

  const int* kernel_shape_data = kernel_shape->cpu_data();
  const int* stride_data = stride->cpu_data();
  const int* pad_data = pad->cpu_data();
  const int* dilation_data = dilation->cpu_data();
  const Dtype* scale_data = scale->cpu_data();
  const unsigned int* remove_center_data = remove_center->cpu_data();
  const unsigned int* remove_bounds_data = remove_bounds->cpu_data();

  // Normalized convolution for the RGBXY filter
  rgbxy_conv_bottom_vec_.clear();
  rgbxy_conv_bottom_vec_.push_back(split_layer_out_blobs_[0].get()); // image
  rgbxy_conv_bottom_vec_.push_back(rgbxy); // embed
  rgbxy_conv_top_vec_.clear();
  rgbxy_conv_top_vec_.push_back(&bilateral_out_blob_);
  LayerParameter rgbxy_layer_param;
  ConvolutionParameter* rgbxy_conv_param =
    rgbxy_layer_param.mutable_convolution_param();
  NormConvParameter* rgbxy_normconv_param =
    rgbxy_layer_param.mutable_norm_conv_param();
  rgbxy_conv_param->add_kernel_size(kernel_shape_data[0]);
  rgbxy_conv_param->add_kernel_size(kernel_shape_data[1]);
  rgbxy_conv_param->add_stride(stride_data[0]);
  rgbxy_conv_param->add_stride(stride_data[1]);
  rgbxy_conv_param->add_pad(pad_data[0]);
  rgbxy_conv_param->add_pad(pad_data[1]);
  rgbxy_conv_param->add_dilation(dilation_data[0]);
  rgbxy_conv_param->add_dilation(dilation_data[1]);
  rgbxy_conv_param->set_num_output(channels_);
  rgbxy_conv_param->set_group(channels_);
  rgbxy_conv_param->mutable_weight_filler()->set_type("constant");
  rgbxy_conv_param->mutable_weight_filler()->set_value(1);
  rgbxy_conv_param->set_bias_term(false);
  rgbxy_normconv_param->set_norm(NormConvParameter_Norm_L2);
  rgbxy_normconv_param->set_scale_term(true);
  rgbxy_normconv_param->mutable_scale_filler()->set_value(scale_data[0]);
  rgbxy_normconv_param->set_remove_center(remove_center_data[0]);
  rgbxy_normconv_param->set_remove_bounds(remove_bounds_data[0]);
  rgbxy_conv_layer_.reset(new NormConvLayer<Dtype>(rgbxy_layer_param));
  rgbxy_conv_layer_->SetUp(rgbxy_conv_bottom_vec_, rgbxy_conv_top_vec_);

  // Normalized convolution for the spatial filter
  xy_conv_bottom_vec_.clear();
  xy_conv_bottom_vec_.push_back(split_layer_out_blobs_[1].get()); // image
  xy_conv_bottom_vec_.push_back(xy); // embed
  xy_conv_top_vec_.clear();
  xy_conv_top_vec_.push_back(&spatial_out_blob_);
  LayerParameter xy_layer_param;
  ConvolutionParameter* xy_conv_param =
    xy_layer_param.mutable_convolution_param();
  NormConvParameter* xy_normconv_param =
    xy_layer_param.mutable_norm_conv_param();
  xy_conv_param->add_kernel_size(kernel_shape_data[2]);
  xy_conv_param->add_kernel_size(kernel_shape_data[3]);
  xy_conv_param->add_stride(stride_data[2]);
  xy_conv_param->add_stride(stride_data[3]);
  xy_conv_param->add_pad(pad_data[2]);
  xy_conv_param->add_pad(pad_data[3]);
  xy_conv_param->add_dilation(dilation_data[2]);
  xy_conv_param->add_dilation(dilation_data[3]);
  xy_conv_param->set_num_output(channels_);
  xy_conv_param->set_group(channels_);
  xy_conv_param->mutable_weight_filler()->set_type("constant");
  xy_conv_param->mutable_weight_filler()->set_value(1);
  xy_conv_param->set_bias_term(false);
  xy_normconv_param->set_norm(NormConvParameter_Norm_L2);
  xy_normconv_param->set_scale_term(true);
  xy_normconv_param->mutable_scale_filler()->set_value(scale_data[1]);
  xy_normconv_param->set_remove_center(remove_center_data[1]);
  xy_normconv_param->set_remove_bounds(remove_bounds_data[1]);
  xy_conv_layer_.reset(new NormConvLayer<Dtype>(xy_layer_param));
  xy_conv_layer_->SetUp(xy_conv_bottom_vec_, xy_conv_top_vec_);


  // LOG(ERROR) << "-------------------------";
  // LOG(ERROR) << "-------------------------";
  // LOG(ERROR) << "kernel_shape  = " 
  // 	     << kernel_shape_data[0] << ", " << kernel_shape_data[1]
  // 	     << ", " << kernel_shape_data[2] << ", " << kernel_shape_data[3];
  // LOG(ERROR) << "stride  = " 
  // 	     << stride_data[0] << ", " << stride_data[1]
  // 	     << ", " << stride_data[2] << ", " << stride_data[3];
  // LOG(ERROR) << "pad  = " 
  // 	     << pad_data[0] << ", " << pad_data[1]
  // 	     << ", " << pad_data[2] << ", " << pad_data[3];
  // LOG(ERROR) << "dilation  = " 
  // 	     << dilation_data[0] << ", " << dilation_data[1]
  // 	     << ", " << dilation_data[2] << ", " << dilation_data[3];
  // LOG(ERROR) << "scale  = " << scale->cpu_data()[0] << ", " << scale->cpu_data()[1];
  // LOG(ERROR) << "remove_center  = " << remove_center_data[0] << ", " << remove_center_data[1];
  // LOG(ERROR) << "remove_bounds  = " << remove_bounds_data[0] << ", " << remove_bounds_data[1];
  // LOG(ERROR) << "-------------------------";
  // LOG(ERROR) << "-------------------------";

}

/**
 * To be invoked before every call to the Forward_cpu() method.
 */
template <typename Dtype>
void NormConvMeanfieldIteration<Dtype>::PrePass(
    const vector<shared_ptr<Blob<Dtype> > >& parameters_to_copy_from) {
    // const vector<shared_ptr<ModifiedPermutohedral> >* const bilateral_lattices,
    // const Blob<Dtype>* const bilateral_norms) {

  // bilateral_lattices_ = bilateral_lattices;
  // bilateral_norms_ = bilateral_norms;

  // Get copies of the up-to-date parameters.
  for (int i = 0; i < parameters_to_copy_from.size(); ++i) {
    blobs_[i]->CopyFrom(*(parameters_to_copy_from[i].get()));
  }
  // LOG(ERROR) << "ok, did prepass  ----------";
}

/**
 * Forward pass during the inference.
 */
template <typename Dtype>
void NormConvMeanfieldIteration<Dtype>::Forward_cpu() {
  //------------------------------- Softmax normalization--------------------
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  //-----------------------------------Message passing-----------------------
  caffe_set(count_, Dtype(0.), message_passing_.mutable_cpu_data());

  split_layer_->Forward(split_layer_bottom_vec_, split_layer_top_vec_);

  // spatial stuff
  xy_conv_layer_->Forward(xy_conv_bottom_vec_, xy_conv_top_vec_);
  for (int n = 0; n < num_; ++n) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_, num_pixels_, channels_, (Dtype) 1.,
        this->blobs_[0]->cpu_data(), spatial_out_blob_.cpu_data() + spatial_out_blob_.offset(n), (Dtype) 0.,
        message_passing_.mutable_cpu_data() + message_passing_.offset(n));
  }
  // bilateral stuff
  rgbxy_conv_layer_->Forward(rgbxy_conv_bottom_vec_, rgbxy_conv_top_vec_);
  for (int n = 0; n < num_; ++n) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_, num_pixels_, channels_, (Dtype) 1.,
        this->blobs_[1]->cpu_data(), bilateral_out_blob_.cpu_data() + bilateral_out_blob_.offset(n), (Dtype) 1.,
        message_passing_.mutable_cpu_data() + message_passing_.offset(n));
  }
  //--------------------------- Compatibility multiplication ----------------
  // Result from message passing needs to be multiplied with compatibility values.
  for (int n = 0; n < num_; ++n) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_, num_pixels_,
        channels_, (Dtype) 1., this->blobs_[2]->cpu_data(),
        message_passing_.cpu_data() + message_passing_.offset(n), (Dtype) 0.,
        pairwise_.mutable_cpu_data() + pairwise_.offset(n));
  }

  //------------------------- Adding unaries, normalization is left to the next iteration --------------
  // Add unary
  sum_layer_->Forward(sum_bottom_vec_, sum_top_vec_);
}

template<typename Dtype>
void NormConvMeanfieldIteration<Dtype>::Backward_cpu() {
  //---------------------------- Add unary gradient --------------------------
  vector<bool> eltwise_propagate_down(2, true);
  sum_layer_->Backward(sum_top_vec_, eltwise_propagate_down, sum_bottom_vec_);

  //---------------------------- Update compatibility diffs ------------------
  caffe_set(this->blobs_[2]->count(), Dtype(0.), this->blobs_[2]->mutable_cpu_diff());

  for (int n = 0; n < num_; ++n) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, channels_, channels_,
                          num_pixels_, (Dtype) 1., pairwise_.cpu_diff() + pairwise_.offset(n),
                          message_passing_.cpu_data() + message_passing_.offset(n), (Dtype) 1.,
                          this->blobs_[2]->mutable_cpu_diff());
  }

  //-------------------------- Gradient after compatibility transform--- -----
  for (int n = 0; n < num_; ++n) {
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, channels_, num_pixels_,
                          channels_, (Dtype) 1., this->blobs_[2]->cpu_data(),
                          pairwise_.cpu_diff() + pairwise_.offset(n), (Dtype) 0.,
                          message_passing_.mutable_cpu_diff() + message_passing_.offset(n));
  }

  // ------------------------- Gradient w.r.t. kernels weights ------------
  caffe_set(this->blobs_[0]->count(), Dtype(0.), this->blobs_[0]->mutable_cpu_diff());
  caffe_set(this->blobs_[1]->count(), Dtype(0.), this->blobs_[1]->mutable_cpu_diff());

  for (int n = 0; n < num_; ++n) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, channels_, channels_,
                          num_pixels_, (Dtype) 1., message_passing_.cpu_diff() + message_passing_.offset(n),
                          spatial_out_blob_.cpu_data() + spatial_out_blob_.offset(n), (Dtype) 1.,
                          this->blobs_[0]->mutable_cpu_diff());
  }

  for (int n = 0; n < num_; ++n) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, channels_, channels_,
                          num_pixels_, (Dtype) 1., message_passing_.cpu_diff() + message_passing_.offset(n),
                          bilateral_out_blob_.cpu_data() + bilateral_out_blob_.offset(n), (Dtype) 1.,
                          this->blobs_[1]->mutable_cpu_diff());
  }

  for (int n = 0; n < num_; ++n) {
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, channels_, num_pixels_, channels_, (Dtype) 1.,
                          this->blobs_[0]->cpu_data(), message_passing_.cpu_diff() + message_passing_.offset(n),
                          (Dtype) 0.,
                          spatial_out_blob_.mutable_cpu_diff() + spatial_out_blob_.offset(n));
  }

  for (int n = 0; n < num_; ++n) {
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, channels_, num_pixels_, channels_, (Dtype) 1.,
                          this->blobs_[1]->cpu_data(), message_passing_.cpu_diff() + message_passing_.offset(n),
                          (Dtype) 0.,
                          bilateral_out_blob_.mutable_cpu_diff() + bilateral_out_blob_.offset(n));
  }


  //--------------------------- Gradient for message passing ---------------
  vector<bool> prop_down(2, true);
  xy_conv_layer_->Backward(xy_conv_top_vec_, prop_down, xy_conv_bottom_vec_);
  rgbxy_conv_layer_->Backward(rgbxy_conv_top_vec_, prop_down, rgbxy_conv_bottom_vec_);

  vector<bool> split_down(1, true);
  split_layer_->Backward(split_layer_top_vec_, split_down, split_layer_bottom_vec_);

  //--------------------------------------------------------------------------------
  vector<bool> propagate_down(2, true);
  softmax_layer_->Backward(softmax_top_vec_, propagate_down, softmax_bottom_vec_);
}

INSTANTIATE_CLASS(NormConvMeanfieldIteration);

}  // namespace caffe

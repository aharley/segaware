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

template <typename Dtype>
void NospConvMeanfieldLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  const caffe::NormConvMeanfieldParameter meanfield_param = this->layer_param_.norm_conv_meanfield_param();

  // LOG(ERROR) << "----------------------";
  // LOG(ERROR) << "ok, doing layersetup";
  // LOG(ERROR) << "----------------------";

  num_iterations_ = meanfield_param.num_iterations();
  // iter_multiplier_ = meanfield_param.iter_multiplier();

  CHECK_GT(num_iterations_, 1) << "Number of iterations must be greater than 1.";
  LOG(INFO) << "This implementation has not been tested batch size > 1.";

  // theta_alpha_ = meanfield_param.theta_alpha();
  // theta_beta_ = meanfield_param.theta_beta();
  // theta_gamma_ = meanfield_param.theta_gamma();

  count_ = bottom[0]->count();
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  num_pixels_ = height_ * width_;

  // LOG(ERROR) << "ok, check it out: ";
  // LOG(ERROR) << "rgbxy count = " << bottom[2]->count();
  // LOG(ERROR) << "rgbxy num = " << bottom[2]->num();
  // LOG(ERROR) << "rgbxy channels = " << bottom[2]->channels();
  // LOG(ERROR) << "rgbxy height = " << bottom[2]->height();
  // LOG(ERROR) << "rgbxy width = " << bottom[2]->width();

  // Setup filter kernel dimensions (kernel_shape_).
  vector<int> normconv_spatial_param_shape(1, 2); // 0,1 for rgbxy
  vector<int> normconv_basic_param_shape(1, 1); // 0 for rgbxy
  int num_spatial_axes_ = 2;
  // Setup kernel dimensions (kernel_shape_).
  kernel_shape_.Reshape(normconv_spatial_param_shape);
  int* kernel_shape_data = kernel_shape_.mutable_cpu_data();
  for (int i = 0; i < num_spatial_axes_; ++i) {
    kernel_shape_data[i] = meanfield_param.kernel_size(0);
    CHECK_GT(kernel_shape_data[i], 0) << "Filter dimensions must be nonzero.";
  }
  // Setup stride dimensions (stride_).
  stride_.Reshape(normconv_spatial_param_shape);
  int* stride_data = stride_.mutable_cpu_data();
  const int num_stride_dims = meanfield_param.stride_size();
  CHECK(num_stride_dims == 0 || num_stride_dims == 1)
    << "stride must be specified once for each norm_conv "
    << "(stride specified " << num_stride_dims << " times).";
  const int kDefaultStride = 1;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    stride_data[i] = (num_stride_dims == 0) ? kDefaultStride :
      meanfield_param.stride(0);
    CHECK_GT(stride_data[i], 0) << "Stride dimensions must be nonzero.";
  }
  // Setup pad dimensions (pad_).
  pad_.Reshape(normconv_spatial_param_shape);
  int* pad_data = pad_.mutable_cpu_data();
  const int num_pad_dims = meanfield_param.pad_size();
  CHECK(num_pad_dims == 0 || num_pad_dims == 1)
    << "pad must be specified once for each norm_conv "
    << "(pad specified " << num_pad_dims << " times).";
  const int kDefaultPad = 0;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    pad_data[i] = (num_pad_dims == 0) ? kDefaultPad :
      meanfield_param.pad(0);
  }
  // Setup dilation dimensions (dilation_).
  dilation_.Reshape(normconv_spatial_param_shape);
  int* dilation_data = dilation_.mutable_cpu_data();
  const int num_dilation_dims = meanfield_param.dilation_size();
  CHECK(num_dilation_dims == 0 || num_dilation_dims == 1)
    << "dilation must be specified once for each norm_conv "
    << "(dilation specified " << num_dilation_dims << " times).";
  const int kDefaultDilation = 0;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    dilation_data[i] = (num_dilation_dims == 0) ? kDefaultDilation :
      meanfield_param.dilation(0);
  }
  // Setup scale terms
  scale_.Reshape(normconv_basic_param_shape);
  Dtype* scale_data = scale_.mutable_cpu_data();
  const int num_scale_dims = meanfield_param.scale_size();
  CHECK(num_scale_dims == 0 || num_scale_dims == 1)
    << "scale must be specified once for each norm_conv "
    << "(scale specified " << num_scale_dims << " times).";
  const int kDefaultScale = 0.1;
  scale_data[0] = (num_scale_dims == 0) ? kDefaultScale :
    meanfield_param.scale(0);
  // Setup remove_center terms
  remove_center_.Reshape(normconv_basic_param_shape);
  unsigned int* remove_center_data = remove_center_.mutable_cpu_data();
  const int num_remove_center_dims = meanfield_param.remove_center_size();
  CHECK(num_remove_center_dims == 0 || num_remove_center_dims == 1)
    << "remove_center must be specified once for each norm_conv "
    << "(remove_center specified " << num_remove_center_dims << " times).";
  const int kDefaultRemoveCenter = 1;
  remove_center_data[0] = (num_remove_center_dims == 0) ? kDefaultRemoveCenter :
    meanfield_param.remove_center(0);
  // Setup remove_bounds terms
  remove_bounds_.Reshape(normconv_basic_param_shape);
  unsigned int* remove_bounds_data = remove_bounds_.mutable_cpu_data();
  const int num_remove_bounds_dims = meanfield_param.remove_bounds_size();
  CHECK(num_remove_bounds_dims == 0 || num_remove_bounds_dims == 1)
    << "remove_bounds must be specified once for each norm_conv "
    << "(remove_bounds specified " << num_remove_bounds_dims << " times).";
  const int kDefaultRemoveBounds = 1;
  remove_bounds_data[0] = (num_remove_bounds_dims == 0) ? kDefaultRemoveBounds :
    meanfield_param.remove_bounds(0);

  // vector<int> weight_shape(2);
  // weight_shape[0] = channels_;
  // weight_shape[1] = 1; // channels_ / group_;
  // for (int i = 0; i < num_spatial_axes_; ++i) {
  //   weight_shape.push_back(kernel_shape_data[i]);
  // }
  // bias_term_ = this->layer_param_.convolution_param().bias_term();
  // vector<int> bias_shape(bias_term_, num_output_);

  top[0]->Reshape(num_, channels_, height_, width_);

  // Initialize the parameters that will updated by backpropagation.
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Multimeanfield layer skipping parameter initialization.";
  } else {
    this->blobs_.resize(2);
    // blobs_[0] - bilateral kernel weights
    // blobs_[1] - compatability matrix

    // Allocate space for kernel weights.
    this->blobs_[0].reset(new Blob<Dtype>(1, 1, channels_, channels_));

    caffe_set(channels_ * channels_, Dtype(0.), this->blobs_[0]->mutable_cpu_data());

    // pFile = fopen("bilateral.par", "r");
    // CHECK(pFile) << "The file 'bilateral.par' is not found. Please create it with initial bilateral kernel weights.";
    for (int i = 0; i < channels_; i++) {
      // fscanf(pFile, "%lf", &this->blobs_[1]->mutable_cpu_data()[i * channels_ + i]);
      // to simplify, let's set these to ones
      this->blobs_[0]->mutable_cpu_data()[i * channels_ + i] = 1;
    }
    // fclose(pFile);

    // Initialize the compatibility matrix to have the Potts model.
    this->blobs_[1].reset(new Blob<Dtype>(1, 1, channels_, channels_));
    caffe_set(channels_ * channels_, Dtype(0.), this->blobs_[1]->mutable_cpu_data());
    for (int c = 0; c < channels_; ++c) {
      (this->blobs_[1]->mutable_cpu_data())[c * channels_ + c] = Dtype(-1.);
    }

    // // Initialize and fill the conv weights:
    // // output channels x input channels per-group x kernel height x kernel width
    // this->blobs_[3].reset(new Blob<Dtype>(weight_shape));
    // shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
    //     this->layer_param_.convolution_param().weight_filler()));
    // weight_filler->Fill(this->blobs_[3].get());
    // // If necessary, initialize and fill the biases.
    // if (bias_term_) {
    //   this->blobs_[4].reset(new Blob<Dtype>(bias_shape));
    //   shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
    //       this->layer_param_.convolution_param().bias_filler()));
    //   bias_filler->Fill(this->blobs_[4].get());
    // }
  }

  // // Initialize the spatial lattice. This does not need to be computed for every image because we use a fixed size.
  // float spatial_kernel[2 * num_pixels_];
  // compute_spatial_kernel(spatial_kernel);
  // spatial_lattice_.reset(new ModifiedPermutohedral());
  // spatial_lattice_->init(spatial_kernel, 2, num_pixels_);

  // // Calculate spatial filter normalization factors.
  // norm_feed_.reset(new Dtype[num_pixels_]);
  // caffe_set(num_pixels_, Dtype(1.0), norm_feed_.get());
  // spatial_norm_.Reshape(1, 1, height_, width_);
  // Dtype* norm_data = spatial_norm_.mutable_cpu_data();
  // spatial_lattice_->compute(norm_data, norm_feed_.get(), 1);
  // for (int i = 0; i < num_pixels_; ++i) {
  //   norm_data[i] = 1.0f / (norm_data[i] + 1e-20f);
  // }

  // // Allocate space for bilateral kernels. This is a temporary buffer used to compute bilateral lattices later.
  // // Also allocate space for holding bilateral filter normalization values.
  // bilateral_kernel_buffer_.reset(new float[5 * num_pixels_]);
  // bilateral_norms_.Reshape(num_, 1, height_, width_);

  // Configure the split layer that is used to make copies of the unary term. One copy for each iteration.
  // ** We do this to have a convenient way to accumulate the gradients from the iterations.
  // It may be possible to optimize this calculation later.
  unary_split_layer_bottom_vec_.clear();
  unary_split_layer_bottom_vec_.push_back(bottom[0]);
  unary_split_layer_top_vec_.clear();
  unary_split_layer_out_blobs_.resize(num_iterations_);
  for (int i = 0; i < num_iterations_; i++) {
    unary_split_layer_out_blobs_[i].reset(new Blob<Dtype>());
    unary_split_layer_top_vec_.push_back(unary_split_layer_out_blobs_[i].get());
  }
  LayerParameter unary_split_layer_param;
  unary_split_layer_.reset(new SplitLayer<Dtype>(unary_split_layer_param));
  unary_split_layer_->SetUp(unary_split_layer_bottom_vec_, unary_split_layer_top_vec_);

  rgbxy_split_layer_bottom_vec_.clear();
  rgbxy_split_layer_bottom_vec_.push_back(bottom[2]);
  rgbxy_split_layer_top_vec_.clear();
  rgbxy_split_layer_out_blobs_.resize(num_iterations_);
  for (int i = 0; i < num_iterations_; i++) {
    rgbxy_split_layer_out_blobs_[i].reset(new Blob<Dtype>());
    rgbxy_split_layer_top_vec_.push_back(rgbxy_split_layer_out_blobs_[i].get());
  }
  LayerParameter rgbxy_split_layer_param;
  rgbxy_split_layer_.reset(new SplitLayer<Dtype>(rgbxy_split_layer_param));
  rgbxy_split_layer_->SetUp(rgbxy_split_layer_bottom_vec_, rgbxy_split_layer_top_vec_);

  // // just so we have something for top, let's put the position image in there
  // caffe_copy(height_*width_*2, bottom[3]->cpu_data(), top[1]->mutable_cpu_data());

  // Make blobs to store outputs of each meanfield iteration. Output of the last iteration is stored in top[0].
  // So we need only (num_iterations_ - 1) blobs.
  iteration_output_blobs_.resize(num_iterations_ - 1);
  for (int i = 0; i < num_iterations_ - 1; ++i) {
    iteration_output_blobs_[i].reset(new Blob<Dtype>(num_, channels_, height_, width_));
  }

  // Make instances of MeanfieldIteration and initialize them.
  meanfield_iterations_.resize(num_iterations_);
  for (int i = 0; i < num_iterations_; ++i) {
    // LOG(ERROR) << "about to init a meanfield alyer";
    meanfield_iterations_[i].reset(new NospConvMeanfieldIteration<Dtype>());
    meanfield_iterations_[i]->OneTimeSetUp(
        unary_split_layer_out_blobs_[i].get(), // unary terms
        (i == 0) ? bottom[1] : iteration_output_blobs_[i - 1].get(), // softmax input
        rgbxy_split_layer_out_blobs_[i].get(), // rgbxy
	&kernel_shape_, &stride_, &pad_, &dilation_, &scale_, // convolution params
	&remove_center_, &remove_bounds_,
        (i == num_iterations_ - 1) ? top[0] : iteration_output_blobs_[i].get()); // output blob
        // spatial_lattice_, // spatial lattice
        // &spatial_norm_); // spatial normalization factors.
  }

  this->param_propagate_down_.resize(this->blobs_.size(), true);
  // this->param_propagate_down_.resize(this->blobs_.size(), false);
  // this->set_param_propagate_down(0,true);
  // this->set_param_propagate_down(1,false);
  // this->set_param_propagate_down(2,false);

  // LOG(INFO) << ("NospConvMeanfieldLayer initialized.");
}

template <typename Dtype>
void NospConvMeanfieldLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	// Do nothing.
}

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
void NospConvMeanfieldLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  // LOG(ERROR) << "pos_count = " << pos_count;
  // // const Dtype* pos_data = bottom[3]->cpu_data();
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
  //   Dtype* norm_output_data = bilateral_norms_.mutable_cpu_data() + bilateral_norms_.offset(n);
  //   bilateral_lattices_[n]->compute(norm_output_data, norm_feed_.get(), 1);
  //   for (int i = 0; i < num_pixels_; ++i) {
  //     norm_output_data[i] = 1.f / (norm_output_data[i] + 1e-20f);
  //   }
  // }

  for (int i = 0; i < num_iterations_; ++i) {
    // LOG(ERROR) << "--------- iteration " << i << " ------------------";
    // meanfield_iterations_[i]->PrePass(this->blobs_, &bilateral_lattices_, &bilateral_norms_);
    meanfield_iterations_[i]->PrePass(this->blobs_);
    meanfield_iterations_[i]->Forward_cpu();
  }
}

/**
 * Backprop through filter-based mean field inference.
 */
template<typename Dtype>
void NospConvMeanfieldLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  for (int i = (num_iterations_ - 1); i >= 0; --i) {
    meanfield_iterations_[i]->Backward_cpu();
  }
  vector<bool> split_layer_propagate_down(1, true);
  unary_split_layer_->Backward(unary_split_layer_top_vec_, split_layer_propagate_down, unary_split_layer_bottom_vec_);
  rgbxy_split_layer_->Backward(rgbxy_split_layer_top_vec_, split_layer_propagate_down, rgbxy_split_layer_bottom_vec_);
  // Accumulate diffs from mean field iters for the parameter blobs.
  for (int blob_id = 0; blob_id < this->blobs_.size(); ++blob_id) {
    Blob<Dtype>* cur_blob = this->blobs_[blob_id].get();
    if (this->param_propagate_down_[blob_id]) {
      caffe_set(cur_blob->count(), Dtype(0), cur_blob->mutable_cpu_diff());
      for (int i = 0; i < num_iterations_; ++i) {
        const Dtype* diffs_to_add = meanfield_iterations_[i]->blobs()[blob_id]->cpu_diff();
        caffe_axpy(cur_blob->count(), Dtype(1.), diffs_to_add, cur_blob->mutable_cpu_diff());
      }
    }
  }
}

INSTANTIATE_CLASS(NospConvMeanfieldLayer);
REGISTER_LAYER_CLASS(NospConvMeanfield);
}  // namespace caffe

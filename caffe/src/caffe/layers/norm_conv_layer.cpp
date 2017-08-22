#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/norm_conv_layer.hpp"
// #include "caffe/util/im2dist.hpp"
// #include "caffe/util/im2col.hpp"

namespace caffe {

template <typename Dtype>
void NormConvLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Configure the kernel size, padding, stride, and inputs.
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  NormConvParameter norm_conv_param = this->layer_param_.norm_conv_param();
  switch (norm_conv_param.norm()) {
  case NormConvParameter_Norm_L1:
    norm_ = 1;
    break;
  case NormConvParameter_Norm_L2:
    norm_ = 2; // really, 0.5*(l2_norm)^2 is implemented
    break;
  default:
    LOG(FATAL) << "Unknown norm (choose L1 or L2)";
  }
  remove_center_ = false;
  if (norm_conv_param.has_remove_center())
    remove_center_ = norm_conv_param.remove_center();
  remove_bounds_ = false;
  if (norm_conv_param.has_remove_bounds())
    remove_bounds_ = norm_conv_param.remove_bounds();
  // Configure the kernel size, padding, stride, and inputs.
  force_nd_im2col_ = conv_param.force_nd_im2col();
  bottom_is_im2col_ = conv_param.bottom_is_im2col();
  channel_axis_ = bottom[0]->CanonicalAxisIndex(conv_param.axis());
  const int first_spatial_axis = channel_axis_ + 1;
  const int num_axes = bottom[0]->num_axes();
  num_spatial_axes_ = num_axes - first_spatial_axis;
  CHECK_GE(num_spatial_axes_, 0);
  vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);
  vector<int> spatial_dim_blob_shape(1, std::max(num_spatial_axes_, 1));
  // Setup filter kernel dimensions (kernel_shape_).
  kernel_shape_.Reshape(spatial_dim_blob_shape);
  int* kernel_shape_data = kernel_shape_.mutable_cpu_data();
  if (conv_param.has_kernel_h() || conv_param.has_kernel_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "kernel_h & kernel_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.kernel_size_size())
        << "Either kernel_size or kernel_h/w should be specified; not both.";
    kernel_shape_data[0] = conv_param.kernel_h();
    kernel_shape_data[1] = conv_param.kernel_w();
  } else {
    const int num_kernel_dims = conv_param.kernel_size_size();
    CHECK(num_kernel_dims == 1 || num_kernel_dims == num_spatial_axes_)
        << "kernel_size must be specified once, or once per spatial dimension "
        << "(kernel_size specified " << num_kernel_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
      for (int i = 0; i < num_spatial_axes_; ++i) {
        kernel_shape_data[i] =
            conv_param.kernel_size((num_kernel_dims == 1) ? 0 : i);
      }
  }
  for (int i = 0; i < num_spatial_axes_; ++i) {
    CHECK_GT(kernel_shape_data[i], 0) << "Filter dimensions must be nonzero.";
  }
  // Setup stride dimensions (stride_).
  stride_.Reshape(spatial_dim_blob_shape);
  int* stride_data = stride_.mutable_cpu_data();
  if (conv_param.has_stride_h() || conv_param.has_stride_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "stride_h & stride_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.stride_size())
        << "Either stride or stride_h/w should be specified; not both.";
    stride_data[0] = conv_param.stride_h();
    stride_data[1] = conv_param.stride_w();
  } else {
    const int num_stride_dims = conv_param.stride_size();
    CHECK(num_stride_dims == 0 || num_stride_dims == 1 ||
          num_stride_dims == num_spatial_axes_)
        << "stride must be specified once, or once per spatial dimension "
        << "(stride specified " << num_stride_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
    const int kDefaultStride = 1;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      stride_data[i] = (num_stride_dims == 0) ? kDefaultStride :
          conv_param.stride((num_stride_dims == 1) ? 0 : i);
      CHECK_GT(stride_data[i], 0) << "Stride dimensions must be nonzero.";
    }
  }
  // Setup pad dimensions (pad_).
  pad_.Reshape(spatial_dim_blob_shape);
  int* pad_data = pad_.mutable_cpu_data();
  if (conv_param.has_pad_h() || conv_param.has_pad_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "pad_h & pad_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.pad_size())
        << "Either pad or pad_h/w should be specified; not both.";
    pad_data[0] = conv_param.pad_h();
    pad_data[1] = conv_param.pad_w();
  } else {
    const int num_pad_dims = conv_param.pad_size();
    CHECK(num_pad_dims == 0 || num_pad_dims == 1 ||
          num_pad_dims == num_spatial_axes_)
        << "pad must be specified once, or once per spatial dimension "
        << "(pad specified " << num_pad_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
    const int kDefaultPad = 0;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      pad_data[i] = (num_pad_dims == 0) ? kDefaultPad :
          conv_param.pad((num_pad_dims == 1) ? 0 : i);
    }
  }
  // Setup dilation dimensions (dilation_).
  dilation_.Reshape(spatial_dim_blob_shape);
  int* dilation_data = dilation_.mutable_cpu_data();
  const int num_dilation_dims = conv_param.dilation_size();
  CHECK(num_dilation_dims == 0 || num_dilation_dims == 1 ||
        num_dilation_dims == num_spatial_axes_)
      << "dilation must be specified once, or once per spatial dimension "
      << "(dilation specified " << num_dilation_dims << " times; "
      << num_spatial_axes_ << " spatial dims).";
  const int kDefaultDilation = 1;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    dilation_data[i] = (num_dilation_dims == 0) ? kDefaultDilation :
                       conv_param.dilation((num_dilation_dims == 1) ? 0 : i);
  }
  // Special case: im2col is the identity for 1x1 convolution with stride 1
  // and no padding, so flag for skipping the buffer and transformation.
  is_1x1_ = true;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    is_1x1_ &=
        kernel_shape_data[i] == 1 && stride_data[i] == 1 && pad_data[i] == 0;
    if (!is_1x1_) { break; }
  }
  // Configure output channels and groups.
  channels_ = bottom[0]->shape(channel_axis_);
  emb_channels_ = bottom[1]->shape(channel_axis_);
  // LOG(ERROR) << "<< img_channels_ = " << channels_;
  // LOG(ERROR) << "<< emb_channels_ = " << emb_channels_;
  num_output_ = this->layer_param_.convolution_param().num_output();
  CHECK_GT(num_output_, 0);
  group_ = this->layer_param_.convolution_param().group();
  CHECK_EQ(channels_ % group_, 0);
  CHECK_EQ(num_output_ % group_, 0)
      << "Number of output should be multiples of group.";
  if (reverse_dimensions()) {
    conv_out_channels_ = channels_;
    conv_in_channels_ = num_output_;
  } else {
    conv_out_channels_ = num_output_;
    conv_in_channels_ = channels_;
  }
  // Handle the parameters: weights and biases.
  // - blobs_[0] holds the filter weights
  // if biases are enabled, they go into blobs_[1]
  // if the scale factor (for normalized convolution) is enabled, it'll either g
  // into blobs[1] or blobs_[2], depending on whether or not biases are used
  vector<int> weight_shape(2);
  weight_shape[0] = conv_out_channels_;
  weight_shape[1] = conv_in_channels_ / group_;
  if (bottom_is_im2col_) {
    for (int i = 0; i < num_spatial_axes_; ++i) {
      weight_shape[1] /= kernel_shape_data[i];
    }
  }
  for (int i = 0; i < num_spatial_axes_; ++i) {
    weight_shape.push_back(kernel_shape_data[i]);
  }
  bias_term_ = this->layer_param_.convolution_param().bias_term();
  vector<int> bias_shape(bias_term_, num_output_);
  scale_term_ = norm_conv_param.scale_term();
  scale_ind_ = bias_term_ ? 2 : 1; // if we have a bias, shift scale_ind_ up
  vector<int> scale_shape(scale_term_, 1);
  if (this->blobs_.size() > 0) {
    CHECK_EQ(1 + bias_term_ + scale_term_, this->blobs_.size())
      << "Incorrect number of weight blobs.";
    if (weight_shape != this->blobs_[0]->shape()) {
      Blob<Dtype> weight_shaped_blob(weight_shape);
      LOG(FATAL) << "Incorrect weight shape: expected shape "
		 << weight_shaped_blob.shape_string() << "; instead, shape was "
		 << this->blobs_[0]->shape_string();
    }
    if (bias_term_ && bias_shape != this->blobs_[1]->shape()) {
      Blob<Dtype> bias_shaped_blob(bias_shape);
      LOG(FATAL) << "Incorrect bias shape: expected shape "
		 << bias_shaped_blob.shape_string() << "; instead, shape was "
		 << this->blobs_[1]->shape_string();
    }
    if (scale_term_ && scale_shape != this->blobs_[scale_ind_]->shape()) {
      Blob<Dtype> scale_shaped_blob(scale_shape);
      LOG(FATAL) << "Incorrect scale shape: expected shape "
		 << scale_shaped_blob.shape_string() << "; instead, shape was "
		 << this->blobs_[scale_ind_]->shape_string();
    }
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_ && scale_term_) {
      this->blobs_.resize(3);
    } else if (bias_term_ || scale_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize and fill the weights:
    // output channels x input channels per-group x kernel height x kernel width
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.convolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the biases.
    if (bias_term_) {
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.convolution_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
    // If necessary, initialize and fill the scale.
    if (scale_term_) {
      this->blobs_[scale_ind_].reset(new Blob<Dtype>(scale_shape));
      shared_ptr<Filler<Dtype> > scale_filler(GetFiller<Dtype>(
          norm_conv_param.scale_filler()));
      scale_filler->Fill(this->blobs_[scale_ind_].get());
    }
  }
  kernel_dim_ = this->blobs_[0]->count(1);
  weight_offset_ = conv_out_channels_ * kernel_dim_ / group_;
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void NormConvLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int first_spatial_axis = channel_axis_ + 1;
  CHECK_EQ(bottom[0]->num_axes(), first_spatial_axis + num_spatial_axes_)
      << "bottom num_axes may not change.";
  num_ = bottom[0]->count(0, channel_axis_);
  CHECK_EQ(bottom[0]->shape(channel_axis_), channels_)
      << "Input size incompatible with convolution kernel.";
  CHECK(bottom[0]->shape(first_spatial_axis) == bottom[1]->shape(first_spatial_axis))
    << "The img and emb must have the same spatial dimensions.";
  CHECK(bottom[0]->shape(first_spatial_axis+1) == bottom[1]->shape(first_spatial_axis+1))
    << "The img and emb must have the same spatial dimensions.";
  // Shape the tops.
  bottom_shape_ = &bottom[0]->shape();
  compute_output_shape();
  vector<int> top_shape(bottom[0]->shape().begin(),
      bottom[0]->shape().begin() + channel_axis_);
  top_shape.push_back(num_output_);
  for (int i = 0; i < num_spatial_axes_; ++i) {
    top_shape.push_back(output_shape_[i]);
  }
  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(top_shape);
  }
  if (reverse_dimensions()) {
    conv_out_spatial_dim_ = bottom[0]->count(first_spatial_axis);
  } else {
    conv_out_spatial_dim_ = top[0]->count(first_spatial_axis);
  }
  col_offset_ = kernel_dim_ * conv_out_spatial_dim_;
  output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_;
  // Setup input dimensions (conv_input_shape_).
  vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);
  conv_input_shape_.Reshape(bottom_dim_blob_shape);
  int* conv_input_shape_data = conv_input_shape_.mutable_cpu_data();
  for (int i = 0; i < num_spatial_axes_ + 1; ++i) {
    if (reverse_dimensions()) {
      conv_input_shape_data[i] = top[0]->shape(channel_axis_ + i);
    } else {
      conv_input_shape_data[i] = bottom[0]->shape(channel_axis_ + i);
    }
  }
  emb_input_shape_.Reshape(bottom_dim_blob_shape);
  int* emb_input_shape_data = emb_input_shape_.mutable_cpu_data();
  for (int i = 0; i < num_spatial_axes_ + 1; ++i) {
    if (reverse_dimensions()) {
      emb_input_shape_data[i] = top[1]->shape(channel_axis_ + i);
    } else {
      emb_input_shape_data[i] = bottom[1]->shape(channel_axis_ + i);
    }
  }
  // The im2col result buffer will only hold one image at a time to avoid
  // overly large memory usage. In the special case of 1x1 convolution
  // it goes lazily unused to save memory.
  col_buffer_shape_.clear();
  col_buffer_shape_.push_back(kernel_dim_ * group_);
  for (int i = 0; i < num_spatial_axes_; ++i) {
    if (reverse_dimensions()) {
      col_buffer_shape_.push_back(input_shape(i + 1));
    } else {
      col_buffer_shape_.push_back(output_shape_[i]);
    }
  }
  col_buffer_.Reshape(col_buffer_shape_);
  emb_col_buffer_shape_.clear();
  diff_col_buffer_shape_.clear();
  sum_buffer_shape_.clear();
  int* kernel_shape_data = kernel_shape_.mutable_cpu_data();
  emb_col_buffer_shape_.push_back(kernel_shape_data[0]*kernel_shape_data[1]);
  diff_col_buffer_shape_.push_back(kernel_shape_data[0]*kernel_shape_data[1]*emb_channels_);
  sum_buffer_shape_.push_back(1);
  for (int i = 0; i < num_spatial_axes_; ++i) {
    emb_col_buffer_shape_.push_back(output_shape_[i]);
    diff_col_buffer_shape_.push_back(output_shape_[i]);
    sum_buffer_shape_.push_back(output_shape_[i]);
  }
  soft_col_buffer_.Reshape(emb_col_buffer_shape_);
  emb_col_buffer_.Reshape(emb_col_buffer_shape_);
  diff_col_buffer_.Reshape(diff_col_buffer_shape_);
  res_col_buffer_.Reshape(col_buffer_shape_);
  emb_bottom_dim_ = bottom[1]->count(channel_axis_);
  sum_buffer_.Reshape(sum_buffer_shape_);
  bottom_dim_ = bottom[0]->count(channel_axis_);
  top_dim_ = top[0]->count(channel_axis_);
  num_kernels_im2col_ = conv_in_channels_ * conv_out_spatial_dim_;
  num_kernels_col2im_ = reverse_dimensions() ? top_dim_ : bottom_dim_;
  // Set up the all ones "bias multiplier" for adding biases by BLAS
  in_spatial_dim_ = bottom[0]->count(first_spatial_axis);
  out_spatial_dim_ = top[0]->count(first_spatial_axis);
  // bias and scale and normalization share this buffer. it's just a vector of ones.
  int sum_mult_size = out_spatial_dim_;
  if (scale_term_)
    sum_mult_size = res_col_buffer_.count();
  vector<int> sum_multiplier_shape(1, sum_mult_size);
  sum_multiplier_.Reshape(sum_multiplier_shape);
  caffe_set(sum_multiplier_.count(), Dtype(1),
	    sum_multiplier_.mutable_cpu_data());
}

template <typename Dtype>
void NormConvLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  const bool bottom_is_im2col = this->bottom_is_im2col_;
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    int output_dim;
    if (!bottom_is_im2col)
      output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    else 
      output_dim = input_dim;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void NormConvLayer<Dtype>::norm_weight_cpu_gemm(const Dtype* output, Dtype* weights) {
  // prep col_buffer_ and emb_col_buffer_
  const int emb_count = emb_col_buffer_.count();
  // multiply the two
  for (int c=0; c < conv_in_channels_; ++c) {
    caffe_mul(emb_count,
	      soft_col_buffer_.cpu_data(),
	      col_buffer_.cpu_data() + c * emb_count,
	      res_col_buffer_.mutable_cpu_data() + c * emb_count);
  }
  // gemm into weights
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
			  kernel_dim_, conv_out_spatial_dim_,
			  (Dtype)1., output + output_offset_ * g,  
			  res_col_buffer_.cpu_data() + col_offset_ * g,
			  (Dtype)1., weights + weight_offset_ * g);
  }
}

template <typename Dtype>
void NormConvLayer<Dtype>::norm_forward_cpu_gemm(const Dtype* weights, 
				      Dtype* output, bool skip_im2col) {
  // prep col_buffer_ and emb_col_buffer_
  const int emb_count = soft_col_buffer_.count();
  // multiply the two
  for (int c=0; c < conv_in_channels_; ++c) {
    caffe_mul(emb_count,
	      soft_col_buffer_.cpu_data(),
	      col_buffer_.cpu_data() + c * emb_count,
	      res_col_buffer_.mutable_cpu_data() + c * emb_count);
  }
  // gemm into output
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ / group_, 
			  conv_out_spatial_dim_, kernel_dim_,
			  (Dtype)1., weights + weight_offset_ * g, 
			  res_col_buffer_.mutable_cpu_data() + col_offset_ * g,
			  (Dtype)0., output + output_offset_ * g);
  }
}

template <typename Dtype>
void NormConvLayer<Dtype>::forward_cpu_bias(Dtype* output,
					    const Dtype* bias) {
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
			out_spatial_dim_, 1, (Dtype)1., bias, sum_multiplier_.cpu_data(),
			(Dtype)1., output);
}

template <typename Dtype>
void NormConvLayer<Dtype>::backward_cpu_bias(Dtype* bias,
					     const Dtype* input) {
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1.,
			input, sum_multiplier_.cpu_data(), 1., bias);
}

template <typename Dtype>
void NormConvLayer<Dtype>::norm_backward_cpu_img_gemm(const Dtype* top_diff,
						      const Dtype* weights, Dtype* input_img) {
  // prep col_buffer_ and emb_col_buffer_
  const int emb_count = emb_col_buffer_.count();
  // gemm into res_col_buffer_
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
  			  conv_out_spatial_dim_, conv_out_channels_ / group_,
  			  (Dtype)1., weights + weight_offset_ * g, top_diff + output_offset_ * g,
  			  (Dtype)0., res_col_buffer_.mutable_cpu_data() + col_offset_ * g);
  }
  // multiply by exp(scale(emb))
  for (int c=0; c < conv_in_channels_; ++c) {
    caffe_mul(emb_count,
  	      soft_col_buffer_.cpu_data(),
  	      res_col_buffer_.cpu_data() + c * emb_count,
  	      res_col_buffer_.mutable_cpu_data() + c * emb_count);
  }
  // col2im
  if (!is_1x1_ && !bottom_is_im2col_) {
    conv_col2im_cpu(res_col_buffer_.cpu_data(), input_img);
  }
}

template <typename Dtype>
void NormConvLayer<Dtype>::norm_backward_cpu_emb_gemm(const Dtype* top_diff,
						   const Dtype* weights, Dtype* emb_diff) {
  // prep col_buffer_ and emb_col_buffer_
  const int img_count = res_col_buffer_.count();
  const int emb_count = emb_col_buffer_.count();
  // gemm into res_col_buffer_
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
			  conv_out_spatial_dim_, conv_out_channels_ / group_,
			  (Dtype)1., weights + weight_offset_ * g, top_diff + output_offset_ * g,
			  (Dtype)0., res_col_buffer_.mutable_cpu_data() + col_offset_ * g);
  }
  // mult by img
  caffe_mul(img_count, 
	    col_buffer_.cpu_data(),
	    res_col_buffer_.cpu_data(),
	    res_col_buffer_.mutable_cpu_data());
  // sum down to one channel
  for (int c=1; c < conv_in_channels_; ++c) {
    caffe_axpy(emb_count,
  	       Dtype(1),
  	       res_col_buffer_.cpu_data() + c * emb_count,
  	       res_col_buffer_.mutable_cpu_data());
  }
  Dtype* sum_data = sum_buffer_.mutable_cpu_data();
  int mask_size = emb_col_buffer_.count(0,channel_axis_);
  // compute dot(top_diff, top_data) and subtract them from the bottom diff
  for (int k = 0; k < conv_out_spatial_dim_; ++k) {
    sum_data[k] = caffe_cpu_strided_dot<Dtype>(mask_size,
	 res_col_buffer_.cpu_data() + k, conv_out_spatial_dim_,
	 soft_col_buffer_.cpu_data() + k, conv_out_spatial_dim_);
  }
  // subtraction
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, mask_size, conv_out_spatial_dim_, 1,
		-1., sum_multiplier_.cpu_data(), sum_data, 1., res_col_buffer_.mutable_cpu_data());
  // elementwise multiplication
  caffe_mul(emb_count, res_col_buffer_.cpu_data(), soft_col_buffer_.cpu_data(), res_col_buffer_.mutable_cpu_data());
  if (scale_term_) {
    // scale the res
    Dtype* scale_factor = this->blobs_[scale_ind_].get()->mutable_cpu_data();
    caffe_cpu_scale(emb_count, 
  		    -scale_factor[0], 
  		    res_col_buffer_.cpu_data(), 
  		    res_col_buffer_.mutable_cpu_data());
  }
  // dist2im
  if (!is_1x1_ && !bottom_is_im2col_) {
    conv_dist2im_cpu(res_col_buffer_.cpu_data(), 
  		     diff_col_buffer_.cpu_data(), 
  		     emb_diff);
  }
}

template <typename Dtype>
void NormConvLayer<Dtype>::prep_buffers_cpu(const Dtype* weights, 
	const Dtype* input_img, const Dtype* input_emb) {
  const int emb_count = emb_col_buffer_.count();
  // get fresh copies of these
  conv_im2col_cpu(input_img, col_buffer_.mutable_cpu_data());
  conv_im2dist_cpu(input_emb, emb_col_buffer_.mutable_cpu_data(), 
		   diff_col_buffer_.mutable_cpu_data());
  // for (int i=0;i<emb_count;i++)
  //   LOG(ERROR) << "emb[" << i << "] = " << emb_col_buffer_.cpu_data()[i];
  // scale the embs
  if (scale_term_) {
    Dtype* scale_factor = this->blobs_[scale_ind_].get()->mutable_cpu_data();
    caffe_cpu_scale(emb_count, -scale_factor[0], 
		    emb_col_buffer_.cpu_data(), 
		    emb_col_buffer_.mutable_cpu_data());
  }
  // softmax...
  // We need to subtract the max to avoid numerical issues, compute the exp,
  // and then normalize.
  Dtype* sum_data = sum_buffer_.mutable_cpu_data();
  int mask_size = emb_col_buffer_.count(0,channel_axis_);
  // initialize sum_data to the first element in each mask
  caffe_copy(conv_out_spatial_dim_, emb_col_buffer_.cpu_data(), sum_data);
  for (int j = 0; j < mask_size; j++) {
    for (int k = 0; k < conv_out_spatial_dim_; k++) {
      sum_data[k] = std::max(sum_data[k],
  			       emb_col_buffer_.cpu_data()[j * conv_out_spatial_dim_ + k]);
    }
  }
  // for (int i=0;i<emb_count;i++)
  //   LOG(ERROR) << "emb[" << i << "] = " << emb_col_buffer_.cpu_data()[i];
  caffe_copy(emb_count, emb_col_buffer_.cpu_data(), soft_col_buffer_.mutable_cpu_data());
  // for (int i=0;i<emb_count;i++)
  //   LOG(ERROR) << "soft[" << i << "] = " << soft_col_buffer_.cpu_data()[i];
  // for (int i=0;i<sum_multiplier_.count();i++)
  //   LOG(ERROR) << "sum[" << i << "] = " << sum_multiplier_.cpu_data()[i];
  // subtraction
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, mask_size, conv_out_spatial_dim_,
  			1, -1., sum_multiplier_.cpu_data(), sum_data, 1., soft_col_buffer_.mutable_cpu_data());
  // exponentiation
  caffe_exp<Dtype>(emb_count, soft_col_buffer_.cpu_data(), soft_col_buffer_.mutable_cpu_data());
  // sum after exp
  caffe_cpu_gemv<Dtype>(CblasTrans, mask_size, conv_out_spatial_dim_, 1.,
  			soft_col_buffer_.cpu_data(), sum_multiplier_.cpu_data(), 0., sum_data);
  // division
  Dtype* soft_col_buff = soft_col_buffer_.mutable_cpu_data();
  for (int j = 0; j < mask_size; j++) {
    caffe_div(conv_out_spatial_dim_, soft_col_buff, sum_data, soft_col_buff);
    soft_col_buff += conv_out_spatial_dim_;
  }
}

template <typename Dtype>
void NormConvLayer<Dtype>::backward_cpu_scale(Dtype* scale_diff,
     const Dtype* weights, const Dtype* input_emb, const Dtype* top_diff) {
  // prep col_buffer_ and emb_col_buffer_
  const int img_count = res_col_buffer_.count();
  const int emb_count = emb_col_buffer_.count();
  // gemm into res_col_buffer_
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
			  conv_out_spatial_dim_, conv_out_channels_ / group_,
			  (Dtype)1., weights + weight_offset_ * g, top_diff + output_offset_ * g,
			  (Dtype)0., res_col_buffer_.mutable_cpu_data() + col_offset_ * g);
  }
  // mult by img
  caffe_mul(img_count, 
	    col_buffer_.cpu_data(),
	    res_col_buffer_.cpu_data(),
	    res_col_buffer_.mutable_cpu_data());
  // sum down to one channel
  for (int c=1; c < conv_in_channels_; ++c) {
    caffe_axpy(emb_count,
  	       Dtype(1),
  	       res_col_buffer_.cpu_data() + c * emb_count,
  	       res_col_buffer_.mutable_cpu_data());
  }
  Dtype* sum_data = sum_buffer_.mutable_cpu_data();
  int mask_size = emb_col_buffer_.count(0,channel_axis_);
  // compute dot(top_diff, top_data) and subtract them from the bottom diff
  for (int k = 0; k < conv_out_spatial_dim_; ++k) {
    sum_data[k] = caffe_cpu_strided_dot<Dtype>(mask_size,
	 res_col_buffer_.cpu_data() + k, conv_out_spatial_dim_,
	 soft_col_buffer_.cpu_data() + k, conv_out_spatial_dim_);
  }
  // subtraction
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, mask_size, conv_out_spatial_dim_, 1,
		-1., sum_multiplier_.cpu_data(), sum_data, 1., res_col_buffer_.mutable_cpu_data());
  // elementwise multiplication
  caffe_mul(emb_count, res_col_buffer_.cpu_data(), soft_col_buffer_.cpu_data(), res_col_buffer_.mutable_cpu_data());
  // get a fresh embdist
  conv_im2dist_cpu(input_emb, emb_col_buffer_.mutable_cpu_data(), 
  		   diff_col_buffer_.mutable_cpu_data());
  // mult by embdist
  caffe_mul(emb_count,
	    emb_col_buffer_.cpu_data(),
	    res_col_buffer_.cpu_data(),
	    res_col_buffer_.mutable_cpu_data());
  // mult by scale sign
  caffe_cpu_scale(emb_count, Dtype(-1), res_col_buffer_.cpu_data(), res_col_buffer_.mutable_cpu_data());
  // add it up
  caffe_cpu_gemv<Dtype>(CblasNoTrans, 1, emb_count, 1.,
			res_col_buffer_.cpu_data(), sum_multiplier_.cpu_data(), 1., scale_diff);
}

// template <typename Dtype>
// void NormConvLayer<Dtype>::norm_backward_cpu_all(const Dtype* top_diff,
//   const Dtype* weights, const Dtype* input_img, const Dtype* input_emb, 
//   Dtype* weight_diff, Dtype* img_diff, Dtype* emb_diff, Dtype* scale_diff) {
//   // doesn't work yet
// }

template <typename Dtype>
void NormConvLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  const Dtype* bottom_img = bottom[0]->cpu_data();
  const Dtype* bottom_emb = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  for (int n = 0; n < this->num_; ++n) {
    this->prep_buffers_cpu(weight, 
  				bottom_img + n * this->bottom_dim_,
  				bottom_emb + n * this->emb_bottom_dim_);
    this->norm_forward_cpu_gemm(weight,
  				top_data + n * this->top_dim_);
    if (this->bias_term_) {
      const Dtype* bias = this->blobs_[1]->cpu_data();
      this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
    }
  }
}

template <typename Dtype>
void NormConvLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* bottom_img = bottom[0]->cpu_data();
  const Dtype* bottom_emb = bottom[1]->cpu_data();
  Dtype* bottom_img_diff = bottom[0]->mutable_cpu_diff();
  Dtype* bottom_emb_diff = bottom[1]->mutable_cpu_diff();

  // Bias gradient, if necessary.
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
    for (int n = 0; n < this->num_; ++n) {
      backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
    }
  }
  if (this->param_propagate_down_[0] 
      || (this->scale_term_ && this->param_propagate_down_[this->scale_ind_])
      || propagate_down[0] || propagate_down[1]) {
    for (int n = 0; n < this->num_; ++n) {
      // doing backprop for all blobs simultaneously is great and all, but too complicated for now. 

      // LOG(ERROR) << "n = " << n;
      // commonly we will want to bprop to everything: weights, image, embeddings, and scale.
      // we can save a bit of time doing these together.
      // if (this->param_propagate_down_[0] && this->scale_term_ && 
      // 	  this->param_propagate_down_[this->scale_ind_] &&
      // 	  propagate_down[0] && propagate_down[1]) {
      // 	Dtype* scale_diff = this->blobs_[this->scale_ind_]->mutable_cpu_diff();
      // 	this->norm_backward_cpu_all(top_diff + n * this->top_dim_, 
      // 					  weight,
      // 					  bottom_img + n * this->bottom_dim_,
      // 					  bottom_emb + n * this->emb_bottom_dim_,
      // 					  weight_diff,
      // 					  bottom_img_diff + n * this->bottom_dim_,
      // 					  bottom_emb_diff + n * this->emb_bottom_dim_,
      // 					  scale_diff);
      // } else {
      // all except scale need a fresh run of im2col and im2dist for data "n"
      if (this->param_propagate_down_[0] || propagate_down[0] || propagate_down[1])
      	prep_buffers_cpu(weight, 
			 bottom_img + n * this->bottom_dim_,
			 bottom_emb + n * this->emb_bottom_dim_);
      // gradient w.r.t. weight. Note that we will accumulate diffs.
      if (this->param_propagate_down_[0])
      	norm_weight_cpu_gemm(top_diff + n * this->top_dim_, weight_diff);
      // gradient w.r.t. bottom data, if necessary.
      if (propagate_down[0])
      	norm_backward_cpu_img_gemm(top_diff + n * this->top_dim_, weight,
				   bottom_img_diff + n * this->bottom_dim_);
      if (propagate_down[1])
      	norm_backward_cpu_emb_gemm(top_diff + n * this->top_dim_, weight,
				   bottom_emb_diff + n * this->emb_bottom_dim_);
      // gradient w.r.t. scale, if necessary
      if (this->scale_term_ && this->param_propagate_down_[this->scale_ind_]) {
      	Dtype* scale_diff = this->blobs_[this->scale_ind_]->mutable_cpu_diff();
      	backward_cpu_scale(scale_diff, weight,
			   bottom_emb + n * this->emb_bottom_dim_,
			   top_diff + n * this->top_dim_);
      }
      // LOG(ERROR) << " done " << n;
      // }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(NormConvLayer);
#endif

INSTANTIATE_CLASS(NormConvLayer);
REGISTER_LAYER_CLASS(NormConv);

}  // namespace caffe

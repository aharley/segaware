#include <vector>

#include "caffe/layers/dist_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void DistLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  DistLossParameter dist_loss_param = this->layer_param_.dist_loss_param();
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  const int input_num_dims = bottom[0]->shape().size();
  channel_axis_ = bottom[0]->CanonicalAxisIndex(conv_param.axis());
  const int first_spatial_dim = channel_axis_ + 1;
  num_spatial_axes_ = input_num_dims - first_spatial_dim;
  CHECK_GE(num_spatial_axes_, 1);
  vector<int> dim_blob_shape(1, num_spatial_axes_);
  // Setup filter kernel dimensions (kernel_shape_).
  kernel_shape_.Reshape(dim_blob_shape);
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
        << num_spatial_axes_ << " spatial dims);";
      for (int i = 0; i < num_spatial_axes_; ++i) {
        kernel_shape_data[i] =
            conv_param.kernel_size((num_kernel_dims == 1) ? 0 : i);
      }
  }
  for (int i = 0; i < num_spatial_axes_; ++i) {
    CHECK_GT(kernel_shape_data[i], 0) << "Filter dimensions must be nonzero.";
  }
  // Setup stride dimensions (stride_).
  stride_.Reshape(dim_blob_shape);
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
        << num_spatial_axes_ << " spatial dims);";
    const int kDefaultStride = 1;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      stride_data[i] = (num_stride_dims == 0) ? kDefaultStride :
          conv_param.stride((num_stride_dims == 1) ? 0 : i);
      CHECK_GT(stride_data[i], 0) << "Stride dimensions must be nonzero.";
    }
  }
  // Setup pad dimensions (pad_).
  pad_.Reshape(dim_blob_shape);
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
        << num_spatial_axes_ << " spatial dims);";
    const int kDefaultPad = 0;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      pad_data[i] = (num_pad_dims == 0) ? kDefaultPad :
          conv_param.pad((num_pad_dims == 1) ? 0 : i);
    }
  }
  // Setup dilation dimensions (dilation_).
  dilation_.Reshape(dim_blob_shape);
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

  has_ignore_label_ = this->layer_param_.dist_loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.dist_loss_param().ignore_label();
  }
  //normalize_ = this->layer_param_.loss_param().normalize();
  
  alpha_ = dist_loss_param.alpha();
  beta_ = dist_loss_param.beta();
  CHECK_GT(alpha_, 0) << "alpha must be greater than 0.";
  CHECK_GT(beta_, 0) << "beta must be greater than 0.";
  CHECK_GT(beta_, alpha_) << "beta must be greater than alpha.";
  //num_output_ = dist_loss_param.num_output(); // deprecated
  // should check to see if these params are entered, and beta > alpha
}

template <typename Dtype>
void DistLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  // CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
  //     << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
  // diff_temp_.ReshapeLike(*bottom[0]);
  // temp_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void DistLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  int num = bottom[0]->num();
  int height_col = bottom[0]->height();
  int width_col = bottom[0]->width();
  int channels_col = bottom[0]->channels();
  int count = bottom[0]->count();
  const Dtype* dist_col = bottom[0]->cpu_data();
  Dtype* parity_col = bottom[1]->mutable_cpu_data();
  Dtype* diff_col = diff_.mutable_cpu_data();

  caffe_set(height_col * width_col * channels_col, Dtype(0), diff_col);
  Dtype loss = 0;
  
  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < channels_col; ++c) {
      for (int h = 0; h < height_col; ++h) {
  	for (int w = 0; w < width_col; ++w) {
  	  Dtype dist = dist_col[((n * channels_col + c) * height_col + h) * width_col + w];
  	  int parity = parity_col[((n * channels_col + c) * height_col + h) * width_col + w];
  	  Dtype offMargin = 0;
	  if (has_ignore_label_ && parity==ignore_label_) {
	    continue;
	  } else {
	    if (parity) {
	      offMargin = std::max(dist-alpha_, Dtype(0));
	      diff_col[((n * channels_col + c) * height_col + h) * width_col + w] = offMargin;
	    } else {
	      offMargin = std::max(beta_-dist, Dtype(0));
	      diff_col[((n * channels_col + c) * height_col + h) * width_col + w] = -offMargin;
	    }
	    loss += offMargin;
	  }
  	}
      }
    }
  }
  top[0]->mutable_cpu_data()[0] = loss / count;
}
  
template <typename Dtype>
void DistLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  int count = bottom[0]->count();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* diff_col = diff_.cpu_data();
  caffe_copy(count,diff_col,bottom_diff);
  caffe_scal(count, Dtype(1) / count, bottom_diff);
}

#ifdef CPU_ONLY
STUB_GPU(DistLossLayer);
#endif

INSTANTIATE_CLASS(DistLossLayer);
REGISTER_LAYER_CLASS(DistLoss);

}  // namespace caffe


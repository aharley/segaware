#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/layers/im2col_layer.hpp"

#ifdef USE_CUDNN
#include "caffe/layers/cudnn_conv_layer.hpp"
#endif

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

// Reference convolution for checking results:
// accumulate through explicit loops over input, output, and filters.
template <typename Dtype>
void caffe_conv(const Blob<Dtype>* in, ConvolutionParameter* conv_param,
    const vector<shared_ptr<Blob<Dtype> > >& weights,
    Blob<Dtype>* out) {
  const bool has_depth = (out->num_axes() == 5);
  if (!has_depth) { CHECK_EQ(4, out->num_axes()); }
  // Kernel size, stride, and pad
  int kernel_h, kernel_w;
  if (conv_param->has_kernel_h() || conv_param->has_kernel_w()) {
    kernel_h = conv_param->kernel_h();
    kernel_w = conv_param->kernel_w();
  } else {
    kernel_h = kernel_w = conv_param->kernel_size(0);
  }
  int pad_h, pad_w;
  if (conv_param->has_pad_h() || conv_param->has_pad_w()) {
    pad_h = conv_param->pad_h();
    pad_w = conv_param->pad_w();
  } else {
    pad_h = pad_w = conv_param->pad_size() ? conv_param->pad(0) : 0;
  }
  int stride_h, stride_w;
  if (conv_param->has_stride_h() || conv_param->has_stride_w()) {
    stride_h = conv_param->stride_h();
    stride_w = conv_param->stride_w();
  } else {
    stride_h = stride_w = conv_param->stride_size() ? conv_param->stride(0) : 1;
  }
  int dilation_h, dilation_w;
  dilation_h = dilation_w = conv_param->dilation_size() ?
                            conv_param->dilation(0) : 1;
  int kernel_d, pad_d, stride_d, dilation_d;
  if (has_depth) {
    kernel_d = kernel_h;
    stride_d = stride_h;
    pad_d = pad_h;
    dilation_d = dilation_h;
  } else {
    kernel_d = stride_d = dilation_d = 1;
    pad_d = 0;
  }
  // Groups
  int groups = conv_param->group();
  int o_g = out->shape(1) / groups;
  int k_g = in->shape(1) / groups;
  int o_head, k_head;
  // Convolution
  vector<int> weight_offset(4 + has_depth);
  vector<int> in_offset(4 + has_depth);
  vector<int> out_offset(4 + has_depth);
  Dtype* out_data = out->mutable_cpu_data();
  for (int n = 0; n < out->shape(0); n++) {
    for (int g = 0; g < groups; g++) {
      o_head = o_g * g;
      k_head = k_g * g;
      for (int o = 0; o < o_g; o++) {
        for (int k = 0; k < k_g; k++) {
          for (int z = 0; z < (has_depth ? out->shape(2) : 1); z++) {
            for (int y = 0; y < out->shape(2 + has_depth); y++) {
              for (int x = 0; x < out->shape(3 + has_depth); x++) {
                for (int r = 0; r < kernel_d; r++) {
                  for (int p = 0; p < kernel_h; p++) {
                    for (int q = 0; q < kernel_w; q++) {
                      int in_z = z * stride_d - pad_d + r * dilation_d;
                      int in_y = y * stride_h - pad_h + p * dilation_h;
                      int in_x = x * stride_w - pad_w + q * dilation_w;
                      if (in_z >= 0 && in_z < (has_depth ? in->shape(2) : 1)
                          && in_y >= 0 && in_y < in->shape(2 + has_depth)
                          && in_x >= 0 && in_x < in->shape(3 + has_depth)) {
                        weight_offset[0] = o + o_head;
                        weight_offset[1] = k;
                        if (has_depth) { weight_offset[2] = r; }
                        weight_offset[2 + has_depth] = p;
                        weight_offset[3 + has_depth] = q;
                        in_offset[0] = n;
                        in_offset[1] = k + k_head;
                        if (has_depth) { in_offset[2] = in_z; }
                        in_offset[2 + has_depth] = in_y;
                        in_offset[3 + has_depth] = in_x;
                        out_offset[0] = n;
                        out_offset[1] = o + o_head;
                        if (has_depth) { out_offset[2] = z; }
                        out_offset[2 + has_depth] = y;
                        out_offset[3 + has_depth] = x;
                        out_data[out->offset(out_offset)] +=
                            in->data_at(in_offset)
                            * weights[0]->data_at(weight_offset);
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  // Bias
  if (conv_param->bias_term()) {
    const Dtype* bias_data = weights[1]->cpu_data();
    for (int n = 0; n < out->shape(0); n++) {
      for (int o = 0; o < out->shape(1); o++) {
        for (int z = 0; z < (has_depth ? out->shape(2) : 1); z++) {
          for (int y = 0; y < out->shape(2 + has_depth); y++) {
            for (int x = 0; x < out->shape(3 + has_depth); x++) {
              out_offset[0] = n;
              out_offset[1] = o;
              if (has_depth) { out_offset[2] = z; }
              out_offset[2 + has_depth] = y;
              out_offset[3 + has_depth] = x;
              out_data[out->offset(out_offset)] += bias_data[o];
            }
          }
        }
      }
    }
  }
}

template void caffe_conv(const Blob<float>* in,
    ConvolutionParameter* conv_param,
    const vector<shared_ptr<Blob<float> > >& weights,
    Blob<float>* out);
template void caffe_conv(const Blob<double>* in,
    ConvolutionParameter* conv_param,
    const vector<shared_ptr<Blob<double> > >& weights,
    Blob<double>* out);

template <typename TypeParam>
class ConvolutionLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ConvolutionLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 6, 4)),
        blob_bottom_2_(new Blob<Dtype>(2, 3, 6, 4)),
        blob_top_(new Blob<Dtype>()),
        blob_top_2_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_value(1.);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    filler.Fill(this->blob_bottom_2_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~ConvolutionLayerTest() {
    delete blob_bottom_;
    delete blob_bottom_2_;
    delete blob_top_;
    delete blob_top_2_;
  }

  virtual Blob<Dtype>* MakeReferenceTop(Blob<Dtype>* top) {
    this->ref_blob_top_.reset(new Blob<Dtype>());
    this->ref_blob_top_->ReshapeLike(*top);
    return this->ref_blob_top_.get();
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_2_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top_2_;
  shared_ptr<Blob<Dtype> > ref_blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(ConvolutionLayerTest, TestDtypesAndDevices);

TYPED_TEST(ConvolutionLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->add_kernel_size(3);
  convolution_param->add_stride(2);
  convolution_param->set_num_output(4);
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  shared_ptr<Layer<Dtype> > layer(
      new ConvolutionLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 4);
  EXPECT_EQ(this->blob_top_->height(), 2);
  EXPECT_EQ(this->blob_top_->width(), 1);
  EXPECT_EQ(this->blob_top_2_->num(), 2);
  EXPECT_EQ(this->blob_top_2_->channels(), 4);
  EXPECT_EQ(this->blob_top_2_->height(), 2);
  EXPECT_EQ(this->blob_top_2_->width(), 1);
  // setting group should not change the shape
  convolution_param->set_num_output(3);
  convolution_param->set_group(3);
  layer.reset(new ConvolutionLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->height(), 2);
  EXPECT_EQ(this->blob_top_->width(), 1);
  EXPECT_EQ(this->blob_top_2_->num(), 2);
  EXPECT_EQ(this->blob_top_2_->channels(), 3);
  EXPECT_EQ(this->blob_top_2_->height(), 2);
  EXPECT_EQ(this->blob_top_2_->width(), 1);
}

TYPED_TEST(ConvolutionLayerTest, TestSimpleConvolution) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->add_kernel_size(3);
  convolution_param->add_stride(2);
  convolution_param->add_pad(1);
  convolution_param->set_num_output(4);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0.1);
  shared_ptr<Layer<Dtype> > layer(
      new ConvolutionLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_conv(this->blob_bottom_, convolution_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
  caffe_conv(this->blob_bottom_2_, convolution_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

// TYPED_TEST(ConvolutionLayerTest, TestNormalizedConvolution) {
//   typedef typename TypeParam::Dtype Dtype;
//   this->blob_bottom_vec_.clear();
//   this->blob_top_vec_.clear();
//   this->blob_bottom_vec_.push_back(this->blob_bottom_); // img
//   this->blob_bottom_vec_.push_back(this->blob_bottom_2_); // emb
//   this->blob_top_vec_.push_back(this->blob_top_);
//   LayerParameter layer_param;
//   ConvolutionParameter* convolution_param =
//       layer_param.mutable_convolution_param();
//   convolution_param->add_kernel_size(3);
//   convolution_param->add_stride(1);
//   convolution_param->add_pad(0);
//   convolution_param->set_num_output(1);
//   // convolution_param->mutable_weight_filler()->set_type("gaussian");
//   // convolution_param->mutable_bias_filler()->set_type("constant");
//   // convolution_param->mutable_bias_filler()->set_value(0.1);
//   convolution_param->set_bias_term(false);

//   vector<int> bottom_shape;
//   bottom_shape.push_back(1);
//   bottom_shape.push_back(1);
//   bottom_shape.push_back(3);
//   bottom_shape.push_back(5);
//   // FillerParameter filler_param;
//   // filler_param.set_value(1.);
//   // GaussianFiller<Dtype> filler(filler_param);
//   // for (int i = 0; i < this->blob_bottom_vec_.size(); ++i) {
//   //   this->blob_bottom_vec_[i]->Reshape(bottom_shape);
//   //   //filler.Fill(this->blob_bottom_vec_[i]);
//   // }
//   this->blob_bottom_vec_[0]->Reshape(bottom_shape);
//   this->blob_bottom_vec_[1]->Reshape(bottom_shape);
//   for (int i = 0; i < 15 * 1 * 1; i += 15) {
//     this->blob_bottom_->mutable_cpu_data()[i +  0] = 1;
//     this->blob_bottom_->mutable_cpu_data()[i +  1] = 2;
//     this->blob_bottom_->mutable_cpu_data()[i +  2] = 5;
//     this->blob_bottom_->mutable_cpu_data()[i +  3] = 2;
//     this->blob_bottom_->mutable_cpu_data()[i +  4] = 3;
//     this->blob_bottom_->mutable_cpu_data()[i +  5] = 9;
//     this->blob_bottom_->mutable_cpu_data()[i +  6] = 1;
//     this->blob_bottom_->mutable_cpu_data()[i +  7] = 1;
//     this->blob_bottom_->mutable_cpu_data()[i +  8] = 1;
//     this->blob_bottom_->mutable_cpu_data()[i +  9] = 8;
//     this->blob_bottom_->mutable_cpu_data()[i + 10] = 1;
//     this->blob_bottom_->mutable_cpu_data()[i + 11] = 2;
//     this->blob_bottom_->mutable_cpu_data()[i + 12] = 5;
//     this->blob_bottom_->mutable_cpu_data()[i + 13] = 2;
//     this->blob_bottom_->mutable_cpu_data()[i + 14] = 3;
//   }
//   for (int i = 0; i < 15 * 1 * 1; i += 15) {
//     this->blob_bottom_2_->mutable_cpu_data()[i +  0] = 1;
//     this->blob_bottom_2_->mutable_cpu_data()[i +  1] = 2;
//     this->blob_bottom_2_->mutable_cpu_data()[i +  2] = 5;
//     this->blob_bottom_2_->mutable_cpu_data()[i +  3] = 2;
//     this->blob_bottom_2_->mutable_cpu_data()[i +  4] = 3;
//     this->blob_bottom_2_->mutable_cpu_data()[i +  5] = 9;
//     this->blob_bottom_2_->mutable_cpu_data()[i +  6] = 1;
//     this->blob_bottom_2_->mutable_cpu_data()[i +  7] = 1;
//     this->blob_bottom_2_->mutable_cpu_data()[i +  8] = 1;
//     this->blob_bottom_2_->mutable_cpu_data()[i +  9] = 8;
//     this->blob_bottom_2_->mutable_cpu_data()[i + 10] = 1;
//     this->blob_bottom_2_->mutable_cpu_data()[i + 11] = 2;
//     this->blob_bottom_2_->mutable_cpu_data()[i + 12] = 5;
//     this->blob_bottom_2_->mutable_cpu_data()[i + 13] = 2;
//     this->blob_bottom_2_->mutable_cpu_data()[i + 14] = 3;
//   }

//   FillerParameter filler_param;
//   GaussianFiller<Dtype> filler(filler_param);
//   Blob<Dtype> top_diff;

//   convolution_param->set_normalized_convolution(true);
//   convolution_param->set_norm(ConvolutionParameter_Norm_L1);
//   ConvolutionLayer<Dtype> layer(layer_param);
//   layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
//   layer.blobs().resize(1);
//   layer.blobs()[0].reset(new Blob<Dtype>(1, 1, 3, 3));
//   Dtype* weights = layer.blobs()[0]->mutable_cpu_data();
//   for (int c = 0; c < 1; ++c) {
//     int i = c * 9;  // 3 x 3 filter
//     weights[i +  0] = -1;
//     weights[i +  1] =  0;
//     weights[i +  2] =  1;
//     weights[i +  3] = -2;
//     weights[i +  4] =  0;
//     weights[i +  5] =  2;
//     weights[i +  6] = -1;
//     weights[i +  7] =  0;
//     weights[i +  8] =  1;
//   }
//   ASSERT_EQ(1, layer.blobs().size());    
//   layer.blobs()[0]->set_cpu_data(weights);

//   layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

//   // // Copy pre-generated top diff into actual top diff;
//   // // do Backward and save result in backward_result.
//   // top_diff.ReshapeLike(*this->blob_top_2_);
//   // filler.Fill(&top_diff);
//   // caffe_copy(top_diff.count(), top_diff.cpu_data(),
//   // 	     this->blob_top_2_->mutable_cpu_diff());

//   // vector<bool> propagate_down(1, true);
//   // ASSERT_EQ(this->blob_top_2_->shape(), top_diff.shape());
//   // caffe_copy(top_diff.count(), top_diff.cpu_data(),
//   // 	     this->blob_top_2_->mutable_cpu_diff());
//   // // first run the bii conv layer backwards
//   // layer.Backward(this->blob_top_vec_, propagate_down,
//   // 		     this->blob_bottom_vec_);
//   // // great! now blob_bottom_blob_2_->cpu_diff() has something for the im2col part
//   // // copy the bii bottom into the i2c top
//   // // get blob_bottom/top_ back
//   // this->blob_bottom_vec_.clear();
//   // this->blob_top_vec_.clear();
//   // this->blob_bottom_vec_.push_back(this->blob_bottom_);
//   // this->blob_top_vec_.push_back(this->blob_top_);
//   // // reshape
//   // for (int i = 0; i < this->blob_top_vec_.size(); ++i) {
//   //   this->blob_top_vec_[i]->ReshapeLike(*this->blob_bottom_2_);
//   // }
//   // caffe_copy(this->blob_bottom_2_->count(), this->blob_bottom_2_->cpu_diff(),
//   // 	     this->blob_top_->mutable_cpu_diff());
//   // layer_im2col.Backward(this->blob_top_vec_, propagate_down,
//   // 			this->blob_bottom_vec_);

//   // Dtype* weight_diffs = layer.blobs()[0]->mutable_cpu_diff();
//   // for (int i = 0; i < layer.blobs()[0]->count(); ++i) {
//   //   LOG(ERROR) << "weight diff[" << i << "] = " 
//   // 	       << weight_diffs[i];
//   // }


//   // typedef typename TypeParam::Dtype Dtype;
//   // this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
//   // this->blob_top_vec_.push_back(this->blob_top_2_);
//   // LayerParameter layer_param;
//   // ConvolutionParameter* convolution_param =
//   //     layer_param.mutable_convolution_param();
//   // convolution_param->add_kernel_size(3);
//   // convolution_param->add_stride(2);
//   // convolution_param->add_pad(1);
//   // convolution_param->set_num_output(4);
//   // convolution_param->mutable_weight_filler()->set_type("gaussian");
//   // convolution_param->mutable_bias_filler()->set_type("constant");
//   // convolution_param->mutable_bias_filler()->set_value(0.1);
//   // shared_ptr<Layer<Dtype> > layer(
//   //     new ConvolutionLayer<Dtype>(layer_param));
//   // layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
//   // layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
//   // // Check against reference convolution.
//   // const Dtype* top_data;
//   // const Dtype* ref_top_data;
//   // caffe_conv(this->blob_bottom_, convolution_param, layer->blobs(),
//   //     this->MakeReferenceTop(this->blob_top_));
//   // top_data = this->blob_top_->cpu_data();
//   // ref_top_data = this->ref_blob_top_->cpu_data();
//   // for (int i = 0; i < this->blob_top_->count(); ++i) {
//   //   EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
//   // }
//   // caffe_conv(this->blob_bottom_2_, convolution_param, layer->blobs(),
//   //     this->MakeReferenceTop(this->blob_top_2_));
//   // top_data = this->blob_top_2_->cpu_data();
//   // ref_top_data = this->ref_blob_top_->cpu_data();
//   // for (int i = 0; i < this->blob_top_->count(); ++i) {
//   //   EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
//   // }
// }

// TYPED_TEST(ConvolutionLayerTest, TestDilatedConvolution) {
//   typedef typename TypeParam::Dtype Dtype;
//   vector<int> bottom_shape;
//   bottom_shape.push_back(2);
//   bottom_shape.push_back(3);
//   bottom_shape.push_back(8);
//   bottom_shape.push_back(7);
//   this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
//   this->blob_top_vec_.push_back(this->blob_top_2_);
//   for (int i = 0; i < this->blob_bottom_vec_.size(); ++i) {
//     this->blob_bottom_vec_[i]->Reshape(bottom_shape);
//   }
//   LayerParameter layer_param;
//   ConvolutionParameter* convolution_param =
//       layer_param.mutable_convolution_param();
//   convolution_param->add_kernel_size(3);
//   convolution_param->add_dilation(2);
//   convolution_param->set_num_output(4);
//   convolution_param->mutable_weight_filler()->set_type("gaussian");
//   convolution_param->mutable_bias_filler()->set_type("constant");
//   convolution_param->mutable_bias_filler()->set_value(0.1);
//   shared_ptr<Layer<Dtype> > layer(
//       new ConvolutionLayer<Dtype>(layer_param));
//   layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
//   layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
//   // Check against reference convolution.
//   const Dtype* top_data;
//   const Dtype* ref_top_data;
//   caffe_conv(this->blob_bottom_, convolution_param, layer->blobs(),
//              this->MakeReferenceTop(this->blob_top_));
//   top_data = this->blob_top_->cpu_data();
//   ref_top_data = this->ref_blob_top_->cpu_data();
//   for (int i = 0; i < this->blob_top_->count(); ++i) {
//     EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
//   }
//   caffe_conv(this->blob_bottom_2_, convolution_param, layer->blobs(),
//              this->MakeReferenceTop(this->blob_top_2_));
//   top_data = this->blob_top_2_->cpu_data();
//   ref_top_data = this->ref_blob_top_->cpu_data();
//   for (int i = 0; i < this->blob_top_->count(); ++i) {
//     EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
//   }
// }

// TYPED_TEST(ConvolutionLayerTest, Test0DConvolution) {
//   typedef typename TypeParam::Dtype Dtype;
//   LayerParameter layer_param;
//   ConvolutionParameter* convolution_param =
//       layer_param.mutable_convolution_param();
//   const int kNumOutput = 3;
//   convolution_param->set_num_output(kNumOutput);
//   convolution_param->set_axis(3);
//   convolution_param->mutable_weight_filler()->set_type("gaussian");
//   convolution_param->mutable_bias_filler()->set_type("gaussian");
//   shared_ptr<Layer<Dtype> > layer(
//       new ConvolutionLayer<Dtype>(layer_param));
//   vector<int> top_shape = this->blob_bottom_->shape();
//   top_shape[3] = kNumOutput;
//   layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
//   EXPECT_EQ(top_shape, this->blob_top_->shape());
//   layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
//   // Check against reference convolution.
//   vector<int> weight_offset(2);
//   const Blob<Dtype>* weight = layer->blobs()[0].get();
//   const Blob<Dtype>* bias = layer->blobs()[1].get();
//   const int num = this->blob_top_->count(3);
//   const int dim = this->blob_top_->shape(3);
//   const int bottom_dim = this->blob_bottom_->shape(3);
//   for (int n = 0; n < num; ++n) {
//     for (int d = 0; d < dim; ++d) {
//       weight_offset[0] = d;
//       Dtype value = bias->cpu_data()[d];
//       for (int bottom_d = 0; bottom_d < bottom_dim; ++bottom_d) {
//         weight_offset[1] = bottom_d;
//         value += weight->data_at(weight_offset) *
//                  this->blob_bottom_->cpu_data()[n * bottom_dim + bottom_d];
//       }
//       EXPECT_NEAR(value, this->blob_top_->cpu_data()[n * dim + d], 1e-4);
//     }
//   }
// }

// TYPED_TEST(ConvolutionLayerTest, TestSimple3DConvolution) {
//   typedef typename TypeParam::Dtype Dtype;
//   this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
//   this->blob_top_vec_.push_back(this->blob_top_2_);
//   vector<int> bottom_shape(5);
//   bottom_shape[0] = this->blob_bottom_vec_[0]->shape(0);
//   bottom_shape[1] = this->blob_bottom_vec_[0]->shape(1);
//   bottom_shape[2] = 5;
//   bottom_shape[3] = this->blob_bottom_vec_[0]->shape(2);
//   bottom_shape[4] = this->blob_bottom_vec_[0]->shape(3);
//   FillerParameter filler_param;
//   GaussianFiller<Dtype> filler(filler_param);
//   for (int i = 0; i < this->blob_bottom_vec_.size(); ++i) {
//     this->blob_bottom_vec_[i]->Reshape(bottom_shape);
//     filler.Fill(this->blob_bottom_vec_[i]);
//   }
//   LayerParameter layer_param;
//   ConvolutionParameter* convolution_param =
//       layer_param.mutable_convolution_param();
//   convolution_param->add_kernel_size(3);
//   convolution_param->add_stride(2);
//   convolution_param->set_num_output(4);
//   convolution_param->mutable_weight_filler()->set_type("gaussian");
//   convolution_param->mutable_bias_filler()->set_type("gaussian");
//   shared_ptr<Layer<Dtype> > layer(
//       new ConvolutionLayer<Dtype>(layer_param));
//   layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
//   layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
//   // Check against reference convolution.
//   const Dtype* top_data;
//   const Dtype* ref_top_data;
//   caffe_conv(this->blob_bottom_, convolution_param, layer->blobs(),
//       this->MakeReferenceTop(this->blob_top_));
//   top_data = this->blob_top_->cpu_data();
//   ref_top_data = this->ref_blob_top_->cpu_data();
//   for (int i = 0; i < this->blob_top_->count(); ++i) {
//     EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
//   }
//   caffe_conv(this->blob_bottom_2_, convolution_param, layer->blobs(),
//       this->MakeReferenceTop(this->blob_top_2_));
//   top_data = this->blob_top_2_->cpu_data();
//   ref_top_data = this->ref_blob_top_->cpu_data();
//   for (int i = 0; i < this->blob_top_->count(); ++i) {
//     EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
//   }
// }

// TYPED_TEST(ConvolutionLayerTest, TestDilated3DConvolution) {
//   typedef typename TypeParam::Dtype Dtype;
//   this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
//   this->blob_top_vec_.push_back(this->blob_top_2_);
//   vector<int> bottom_shape(5);
//   bottom_shape[0] = this->blob_bottom_vec_[0]->shape(0);
//   bottom_shape[1] = this->blob_bottom_vec_[0]->shape(1);
//   bottom_shape[2] = 6;
//   bottom_shape[3] = 7;
//   bottom_shape[4] = 8;
//   FillerParameter filler_param;
//   GaussianFiller<Dtype> filler(filler_param);
//   for (int i = 0; i < this->blob_bottom_vec_.size(); ++i) {
//     this->blob_bottom_vec_[i]->Reshape(bottom_shape);
//     filler.Fill(this->blob_bottom_vec_[i]);
//   }
//   LayerParameter layer_param;
//   ConvolutionParameter* convolution_param =
//       layer_param.mutable_convolution_param();
//   convolution_param->add_kernel_size(3);
//   convolution_param->add_dilation(2);
//   convolution_param->set_num_output(4);
//   convolution_param->mutable_weight_filler()->set_type("gaussian");
//   convolution_param->mutable_bias_filler()->set_type("gaussian");
//   shared_ptr<Layer<Dtype> > layer(
//       new ConvolutionLayer<Dtype>(layer_param));
//   layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
//   layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
//   // Check against reference convolution.
//   const Dtype* top_data;
//   const Dtype* ref_top_data;
//   caffe_conv(this->blob_bottom_, convolution_param, layer->blobs(),
//              this->MakeReferenceTop(this->blob_top_));
//   top_data = this->blob_top_->cpu_data();
//   ref_top_data = this->ref_blob_top_->cpu_data();
//   for (int i = 0; i < this->blob_top_->count(); ++i) {
//     EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
//   }
//   caffe_conv(this->blob_bottom_2_, convolution_param, layer->blobs(),
//              this->MakeReferenceTop(this->blob_top_2_));
//   top_data = this->blob_top_2_->cpu_data();
//   ref_top_data = this->ref_blob_top_->cpu_data();
//   for (int i = 0; i < this->blob_top_->count(); ++i) {
//     EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
//   }
// }

// TYPED_TEST(ConvolutionLayerTest, Test1x1Convolution) {
//   typedef typename TypeParam::Dtype Dtype;
//   LayerParameter layer_param;
//   ConvolutionParameter* convolution_param =
//       layer_param.mutable_convolution_param();
//   convolution_param->add_kernel_size(1);
//   convolution_param->add_stride(1);
//   convolution_param->set_num_output(4);
//   convolution_param->mutable_weight_filler()->set_type("gaussian");
//   convolution_param->mutable_bias_filler()->set_type("constant");
//   convolution_param->mutable_bias_filler()->set_value(0.1);
//   shared_ptr<Layer<Dtype> > layer(
//       new ConvolutionLayer<Dtype>(layer_param));
//   layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
//   layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
//   // Check against reference convolution.
//   const Dtype* top_data;
//   const Dtype* ref_top_data;
//   caffe_conv(this->blob_bottom_, convolution_param, layer->blobs(),
//       this->MakeReferenceTop(this->blob_top_));
//   top_data = this->blob_top_->cpu_data();
//   ref_top_data = this->ref_blob_top_->cpu_data();
//   for (int i = 0; i < this->blob_top_->count(); ++i) {
//     EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
//   }
// }

// TYPED_TEST(ConvolutionLayerTest, TestSimpleConvolutionGroup) {
//   typedef typename TypeParam::Dtype Dtype;
//   LayerParameter layer_param;
//   ConvolutionParameter* convolution_param =
//       layer_param.mutable_convolution_param();
//   convolution_param->add_kernel_size(3);
//   convolution_param->add_stride(2);
//   convolution_param->set_num_output(3);
//   convolution_param->set_group(3);
//   convolution_param->mutable_weight_filler()->set_type("gaussian");
//   convolution_param->mutable_bias_filler()->set_type("constant");
//   convolution_param->mutable_bias_filler()->set_value(0.1);
//   shared_ptr<Layer<Dtype> > layer(
//       new ConvolutionLayer<Dtype>(layer_param));
//   layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
//   layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
//   // Check against reference convolution.
//   const Dtype* top_data;
//   const Dtype* ref_top_data;
//   caffe_conv(this->blob_bottom_, convolution_param, layer->blobs(),
//       this->MakeReferenceTop(this->blob_top_));
//   top_data = this->blob_top_->cpu_data();
//   ref_top_data = this->ref_blob_top_->cpu_data();
//   for (int i = 0; i < this->blob_top_->count(); ++i) {
//     EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
//   }
// }

// TYPED_TEST(ConvolutionLayerTest, TestSobelConvolution) {
//   // Test separable convolution by computing the Sobel operator
//   // as a single filter then comparing the result
//   // as the convolution of two rectangular filters.
//   typedef typename TypeParam::Dtype Dtype;
//   // Fill bottoms with identical Gaussian noise.
//   shared_ptr<GaussianFiller<Dtype> > filler;
//   FillerParameter filler_param;
//   filler_param.set_value(1.);
//   filler.reset(new GaussianFiller<Dtype>(filler_param));
//   filler->Fill(this->blob_bottom_);
//   this->blob_bottom_2_->CopyFrom(*this->blob_bottom_);
//   // Compute Sobel G_x operator as 3 x 3 convolution.
//   LayerParameter layer_param;
//   ConvolutionParameter* convolution_param =
//       layer_param.mutable_convolution_param();
//   convolution_param->add_kernel_size(3);
//   convolution_param->add_stride(2);
//   convolution_param->set_num_output(1);
//   convolution_param->set_bias_term(false);
//   shared_ptr<Layer<Dtype> > layer(
//       new ConvolutionLayer<Dtype>(layer_param));
//   layer->blobs().resize(1);
//   layer->blobs()[0].reset(new Blob<Dtype>(1, 3, 3, 3));
//   Dtype* weights = layer->blobs()[0]->mutable_cpu_data();
//   for (int c = 0; c < 3; ++c) {
//     int i = c * 9;  // 3 x 3 filter
//     weights[i +  0] = -1;
//     weights[i +  1] =  0;
//     weights[i +  2] =  1;
//     weights[i +  3] = -2;
//     weights[i +  4] =  0;
//     weights[i +  5] =  2;
//     weights[i +  6] = -1;
//     weights[i +  7] =  0;
//     weights[i +  8] =  1;
//   }
//   layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
//   layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
//   // Compute Sobel G_x operator as separable 3 x 1 and 1 x 3 convolutions.
//   // (1) the [1 2 1] column filter
//   vector<Blob<Dtype>*> sep_blob_bottom_vec;
//   vector<Blob<Dtype>*> sep_blob_top_vec;
//   shared_ptr<Blob<Dtype> > blob_sep(new Blob<Dtype>());
//   sep_blob_bottom_vec.push_back(this->blob_bottom_2_);
//   sep_blob_top_vec.push_back(this->blob_top_2_);
//   convolution_param->clear_kernel_size();
//   convolution_param->clear_stride();
//   convolution_param->set_kernel_h(3);
//   convolution_param->set_kernel_w(1);
//   convolution_param->set_stride_h(2);
//   convolution_param->set_stride_w(1);
//   convolution_param->set_num_output(1);
//   convolution_param->set_bias_term(false);
//   layer.reset(new ConvolutionLayer<Dtype>(layer_param));
//   layer->blobs().resize(1);
//   layer->blobs()[0].reset(new Blob<Dtype>(1, 3, 3, 1));
//   Dtype* weights_1 = layer->blobs()[0]->mutable_cpu_data();
//   for (int c = 0; c < 3; ++c) {
//     int i = c * 3;  // 3 x 1 filter
//     weights_1[i +  0] = 1;
//     weights_1[i +  1] = 2;
//     weights_1[i +  2] = 1;
//   }
//   layer->SetUp(sep_blob_bottom_vec, sep_blob_top_vec);
//   layer->Forward(sep_blob_bottom_vec, sep_blob_top_vec);
//   // (2) the [-1 0 1] row filter
//   blob_sep->CopyFrom(*this->blob_top_2_, false, true);
//   sep_blob_bottom_vec.clear();
//   sep_blob_bottom_vec.push_back(blob_sep.get());
//   convolution_param->set_kernel_h(1);
//   convolution_param->set_kernel_w(3);
//   convolution_param->set_stride_h(1);
//   convolution_param->set_stride_w(2);
//   convolution_param->set_num_output(1);
//   convolution_param->set_bias_term(false);
//   layer.reset(new ConvolutionLayer<Dtype>(layer_param));
//   layer->blobs().resize(1);
//   layer->blobs()[0].reset(new Blob<Dtype>(1, 1, 1, 3));
//   Dtype* weights_2 = layer->blobs()[0]->mutable_cpu_data();
//   weights_2[0] = -1;
//   weights_2[1] =  0;
//   weights_2[2] =  1;
//   layer->SetUp(sep_blob_bottom_vec, sep_blob_top_vec);
//   layer->Forward(sep_blob_bottom_vec, sep_blob_top_vec);
//   // Test equivalence of full and separable filters.
//   const Dtype* top_data = this->blob_top_->cpu_data();
//   const Dtype* sep_top_data = this->blob_top_2_->cpu_data();
//   for (int i = 0; i < this->blob_top_->count(); ++i) {
//     EXPECT_NEAR(top_data[i], sep_top_data[i], 1e-4);
//   }
// }

TYPED_TEST(ConvolutionLayerTest, TestNDAgainst2D) {
  typedef typename TypeParam::Dtype Dtype;
  const int kernel_h = 11;
  const int kernel_w = 13;
  vector<int> bottom_shape(4);
  bottom_shape[0] = 15;
  bottom_shape[1] = 18;
  bottom_shape[2] = kernel_h * 2;
  bottom_shape[3] = kernel_w * 2;
  FillerParameter filler_param;
  GaussianFiller<Dtype> filler(filler_param);
  for (int i = 0; i < this->blob_bottom_vec_.size(); ++i) {
    this->blob_bottom_vec_[i]->Reshape(bottom_shape);
    filler.Fill(this->blob_bottom_vec_[i]);
  }
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->set_num_output(12);
  convolution_param->set_bias_term(false);
  convolution_param->set_group(6);
  convolution_param->set_kernel_h(kernel_h);
  convolution_param->set_kernel_w(kernel_w);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  Blob<Dtype> weights;
  Blob<Dtype> top_diff;
  // Shape and fill weights and top_diff.
  bool copy_diff;
  bool reshape;
  {
    ConvolutionLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    top_diff.ReshapeLike(*this->blob_top_);
    filler.Fill(&top_diff);
    ASSERT_EQ(1, layer.blobs().size());
    copy_diff = false; reshape = true;
    weights.CopyFrom(*layer.blobs()[0], copy_diff, reshape);
  }
  vector<bool> propagate_down(1, true);
  Blob<Dtype> result_2d;
  Blob<Dtype> backward_result_2d;
  Blob<Dtype> backward_weight_result_2d;
  // Test with 2D im2col
  {
    caffe_set(this->blob_top_->count(), Dtype(0),
              this->blob_top_->mutable_cpu_data());
    caffe_set(this->blob_bottom_->count(), Dtype(0),
              this->blob_bottom_->mutable_cpu_diff());
    caffe_set(weights.count(), Dtype(0), weights.mutable_cpu_diff());
    // Do SetUp and Forward; save Forward result in result_2d.
    convolution_param->set_force_nd_im2col(false);
    ConvolutionLayer<Dtype> layer_2d(layer_param);
    layer_2d.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    ASSERT_EQ(1, layer_2d.blobs().size());
    copy_diff = false; reshape = false;
    layer_2d.blobs()[0]->CopyFrom(weights, copy_diff, reshape);
    layer_2d.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    copy_diff = false; reshape = true;
    result_2d.CopyFrom(*this->blob_top_, copy_diff, reshape);
    // Copy pre-generated top diff into actual top diff;
    // do Backward and save result in backward_result_2d.
    ASSERT_EQ(this->blob_top_->shape(), top_diff.shape());
    caffe_copy(top_diff.count(), top_diff.cpu_data(),
               this->blob_top_->mutable_cpu_diff());
    layer_2d.Backward(this->blob_top_vec_, propagate_down,
                      this->blob_bottom_vec_);
    copy_diff = true; reshape = true;
    backward_result_2d.CopyFrom(*this->blob_bottom_, copy_diff, reshape);
    //backward_weight_result_2d.CopyFrom(weights, copy_diff, reshape);
    backward_weight_result_2d.CopyFrom(*layer_2d.blobs()[0], copy_diff, reshape);
  }
  Blob<Dtype> result_nd;
  Blob<Dtype> backward_result_nd;
  Blob<Dtype> backward_weight_result_nd;
  // Test with ND im2col
  {
    caffe_set(this->blob_top_->count(), Dtype(0),
              this->blob_top_->mutable_cpu_data());
    caffe_set(this->blob_bottom_->count(), Dtype(0),
              this->blob_bottom_->mutable_cpu_diff());
    caffe_set(weights.count(), Dtype(0), weights.mutable_cpu_diff());
    // Do SetUp and Forward; save Forward result in result_nd.
    convolution_param->set_force_nd_im2col(true);
    ConvolutionLayer<Dtype> layer_nd(layer_param);
    layer_nd.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    ASSERT_EQ(1, layer_nd.blobs().size());
    copy_diff = false; reshape = false;
    layer_nd.blobs()[0]->CopyFrom(weights, copy_diff, reshape);
    layer_nd.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    copy_diff = false; reshape = true;
    result_nd.CopyFrom(*this->blob_top_, copy_diff, reshape);
    // Copy pre-generated top diff into actual top diff;
    // do Backward and save result in backward_result_nd.
    ASSERT_EQ(this->blob_top_->shape(), top_diff.shape());
    caffe_copy(top_diff.count(), top_diff.cpu_data(),
               this->blob_top_->mutable_cpu_diff());
    layer_nd.Backward(this->blob_top_vec_, propagate_down,
                      this->blob_bottom_vec_);
    copy_diff = true; reshape = true;
    backward_result_nd.CopyFrom(*this->blob_bottom_, copy_diff, reshape);
    // backward_weight_result_nd.CopyFrom(weights, copy_diff, reshape);
    backward_weight_result_nd.CopyFrom(*layer_nd.blobs()[0], copy_diff, reshape);
  }
  ASSERT_EQ(result_nd.count(), result_2d.count());
  for (int i = 0; i < result_2d.count(); ++i)  {
    EXPECT_EQ(result_2d.cpu_data()[i], result_nd.cpu_data()[i]);
  }
  ASSERT_EQ(backward_result_nd.count(), backward_result_2d.count());
  for (int i = 0; i < backward_result_2d.count(); ++i) {
    EXPECT_EQ(backward_result_2d.cpu_diff()[i],
              backward_result_nd.cpu_diff()[i]);
  }
  ASSERT_EQ(backward_weight_result_nd.count(),
            backward_weight_result_2d.count());
  for (int i = 0; i < backward_weight_result_2d.count(); ++i) {
    EXPECT_EQ(backward_weight_result_2d.cpu_diff()[i],
              backward_weight_result_nd.cpu_diff()[i]);
  }
  // caffe_copy(top_diff.count(), top_diff.cpu_data(),
  //            this->blob_top_->mutable_cpu_diff());
  // layer_nd.Backward(this->blob_top_vec_, propagate_down,
  //                   this->blob_bottom_vec_);
  // for (int i = 0; i < this->blob_bottom_->count(); ++i) {
  //   LOG(ERROR) << "blob_bot[" << i << "] = " << this->blob_bottom_->cpu_diff()[i];
  // }
  // for (int i = 0; i < backward_result_nd.count(); ++i) {
  //   LOG(ERROR) << "backward_result_nd[" << i << "] = " << backward_result_nd.cpu_diff()[i];
  // }
    // for (int i = 0; i < backward_weight_result_nd.count(); ++i) {
    //   LOG(ERROR) << "backward_weight_result_nd[" << i << "] = " << backward_weight_result_nd.cpu_diff()[i];
    //   // LOG(ERROR) << "backward_weight_result_nd[" << i << "] = " << layer_nd.blobs()[0]->cpu_diff()[i];


    // }

}

TYPED_TEST(ConvolutionLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  convolution_param->add_kernel_size(3);
  convolution_param->add_stride(2);
  convolution_param->set_num_output(2);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  ConvolutionLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(ConvolutionLayerTest, TestBottomIsIm2col) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.clear();
  this->blob_top_vec_.clear();
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  this->blob_top_vec_.push_back(this->blob_top_);
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->add_kernel_size(3);
  convolution_param->add_stride(1);
  convolution_param->add_pad(0);
  convolution_param->set_num_output(1);
  // convolution_param->mutable_weight_filler()->set_type("gaussian");
  // convolution_param->mutable_bias_filler()->set_type("constant");
  // convolution_param->mutable_bias_filler()->set_value(0.1);
  convolution_param->set_bias_term(false);

  vector<int> bottom_shape;
  bottom_shape.push_back(1);
  bottom_shape.push_back(1);
  bottom_shape.push_back(3);
  bottom_shape.push_back(5);
  // FillerParameter filler_param;
  // filler_param.set_value(1.);
  // GaussianFiller<Dtype> filler(filler_param);
  // for (int i = 0; i < this->blob_bottom_vec_.size(); ++i) {
  //   this->blob_bottom_vec_[i]->Reshape(bottom_shape);
  //   //filler.Fill(this->blob_bottom_vec_[i]);
  // }
  this->blob_bottom_vec_[0]->Reshape(bottom_shape);
  for (int i = 0; i < 15 * 1 * 1; i += 15) {
    this->blob_bottom_->mutable_cpu_data()[i +  0] = 1;
    this->blob_bottom_->mutable_cpu_data()[i +  1] = 2;
    this->blob_bottom_->mutable_cpu_data()[i +  2] = 5;
    this->blob_bottom_->mutable_cpu_data()[i +  3] = 2;
    this->blob_bottom_->mutable_cpu_data()[i +  4] = 3;
    this->blob_bottom_->mutable_cpu_data()[i +  5] = 9;
    this->blob_bottom_->mutable_cpu_data()[i +  6] = 1;
    this->blob_bottom_->mutable_cpu_data()[i +  7] = 1;
    this->blob_bottom_->mutable_cpu_data()[i +  8] = 1;
    this->blob_bottom_->mutable_cpu_data()[i +  9] = 8;
    this->blob_bottom_->mutable_cpu_data()[i + 10] = 1;
    this->blob_bottom_->mutable_cpu_data()[i + 11] = 2;
    this->blob_bottom_->mutable_cpu_data()[i + 12] = 5;
    this->blob_bottom_->mutable_cpu_data()[i + 13] = 2;
    this->blob_bottom_->mutable_cpu_data()[i + 14] = 3;
  }


  bool do_conv = false;

  FillerParameter filler_param;
  GaussianFiller<Dtype> filler(filler_param);
  Blob<Dtype> top_diff;

  // bool copy_diff;
  // bool reshape;

  if (do_conv) {
    shared_ptr<Layer<Dtype> > layer(new ConvolutionLayer<Dtype>(layer_param));

    layer->blobs().resize(1);
    layer->blobs()[0].reset(new Blob<Dtype>(1, 1, 3, 3));
    Dtype* weights = layer->blobs()[0]->mutable_cpu_data();
    for (int c = 0; c < 1; ++c) {
      int i = c * 9;  // 3 x 3 filter
      weights[i +  0] = -1;
      weights[i +  1] =  0;
      weights[i +  2] =  1;
      weights[i +  3] = -2;
      weights[i +  4] =  0;
      weights[i +  5] =  2;
      weights[i +  6] = -1;
      weights[i +  7] =  0;
      weights[i +  8] =  1;
    }

    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    const Dtype* top_data;
    top_data = this->blob_top_->cpu_data();
    for (int i = 0; i < this->blob_top_->count(); ++i) {
      LOG(ERROR) << "conv top_data[" << i << "] = " 
		 << top_data[i];
    }

    top_diff.ReshapeLike(*this->blob_top_);
    filler.Fill(&top_diff);
    caffe_copy(top_diff.count(), top_diff.cpu_data(),
	       this->blob_top_->mutable_cpu_diff());

    vector<bool> propagate_down(1, true);
    layer->Backward(this->blob_top_vec_, propagate_down,
		    this->blob_bottom_vec_);


    Dtype* weight_diffs = layer->blobs()[0]->mutable_cpu_diff();
    for (int i = 0; i < layer->blobs()[0]->count(); ++i) {
      LOG(ERROR) << "weight diff[" << i << "] = " 
		 << weight_diffs[i];
    }
  } else { // do im2col conv_bii
    
    shared_ptr<Layer<Dtype> > layer(new Im2colLayer<Dtype>(layer_param));

    this->blob_bottom_vec_.clear();
    this->blob_top_vec_.clear();
    this->blob_bottom_vec_.push_back(this->blob_bottom_);
    this->blob_top_vec_.push_back(this->blob_top_);
    Im2colLayer<Dtype> layer_im2col(layer_param);
    layer_im2col.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    caffe_set(this->blob_top_->count(), Dtype(0),
    	      this->blob_top_->mutable_cpu_data());
    caffe_set(this->blob_bottom_->count(), Dtype(0),
    	      this->blob_bottom_->mutable_cpu_diff());

    layer_im2col.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    // const Dtype* top_data;
    // top_data = this->blob_top_->cpu_data();
    // for (int i = 0; i < this->blob_top_->count(); ++i) {
    //   LOG(ERROR) << "im2col top_data[" << i << "] = " 
    // 		 << top_data[i];
    // }

    // do conv with bottom_is_im2col, using blob_bottom/top_2_
    // reset, and set up blob_bottom/top_2_
    this->blob_bottom_vec_.clear();
    this->blob_top_vec_.clear();
    this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
    this->blob_top_vec_.push_back(this->blob_top_2_);
    // reshape the bottom like the im2col top
    for (int i = 0; i < this->blob_bottom_vec_.size(); ++i) {
      this->blob_bottom_vec_[i]->ReshapeLike(*this->blob_top_);
    }
    convolution_param->set_bottom_is_im2col(true);
    ConvolutionLayer<Dtype> layer_bii(layer_param);
    layer_bii.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer_bii.blobs().resize(1);
    layer_bii.blobs()[0].reset(new Blob<Dtype>(1, 1, 3, 3));
    Dtype* weights = layer_bii.blobs()[0]->mutable_cpu_data();
    for (int c = 0; c < 1; ++c) {
      int i = c * 9;  // 3 x 3 filter
      weights[i +  0] = -1;
      weights[i +  1] =  0;
      weights[i +  2] =  1;
      weights[i +  3] = -2;
      weights[i +  4] =  0;
      weights[i +  5] =  2;
      weights[i +  6] = -1;
      weights[i +  7] =  0;
      weights[i +  8] =  1;
    }

    caffe_set(this->blob_top_2_->count(), Dtype(0),
    	      this->blob_top_2_->mutable_cpu_data());
    caffe_set(this->blob_bottom_2_->count(), Dtype(0),
    	      this->blob_bottom_2_->mutable_cpu_diff());

    // copy the im2col out into bottom_blob_2_
    caffe_copy(this->blob_top_->count(), this->blob_top_->cpu_data(),
    	       this->blob_bottom_2_->mutable_cpu_data());

    ASSERT_EQ(1, layer_bii.blobs().size());    
    layer_bii.blobs()[0]->set_cpu_data(weights);
    layer_bii.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    // Copy pre-generated top diff into actual top diff;
    // do Backward and save result in backward_result_bii.
    top_diff.ReshapeLike(*this->blob_top_2_);
    filler.Fill(&top_diff);
    caffe_copy(top_diff.count(), top_diff.cpu_data(),
    	       this->blob_top_2_->mutable_cpu_diff());

    vector<bool> propagate_down(1, true);
    ASSERT_EQ(this->blob_top_2_->shape(), top_diff.shape());
    caffe_copy(top_diff.count(), top_diff.cpu_data(),
    	       this->blob_top_2_->mutable_cpu_diff());
    // first run the bii conv layer backwards
    layer_bii.Backward(this->blob_top_vec_, propagate_down,
    		       this->blob_bottom_vec_);
    // great! now blob_bottom_blob_2_->cpu_diff() has something for the im2col part
    // copy the bii bottom into the i2c top
    // get blob_bottom/top_ back
    this->blob_bottom_vec_.clear();
    this->blob_top_vec_.clear();
    this->blob_bottom_vec_.push_back(this->blob_bottom_);
    this->blob_top_vec_.push_back(this->blob_top_);
    // reshape
    for (int i = 0; i < this->blob_top_vec_.size(); ++i) {
      this->blob_top_vec_[i]->ReshapeLike(*this->blob_bottom_2_);
    }
    caffe_copy(this->blob_bottom_2_->count(), this->blob_bottom_2_->cpu_diff(),
    	       this->blob_top_->mutable_cpu_diff());
    layer_im2col.Backward(this->blob_top_vec_, propagate_down,
    			  this->blob_bottom_vec_);

    Dtype* weight_diffs = layer_bii.blobs()[0]->mutable_cpu_diff();
    for (int i = 0; i < layer_bii.blobs()[0]->count(); ++i) {
      LOG(ERROR) << "weight diff[" << i << "] = " 
		 << weight_diffs[i];
    }
  }
}


TYPED_TEST(ConvolutionLayerTest, TestBottomIsIm2colGradient) {
  typedef typename TypeParam::Dtype Dtype;
  const int kernel_h = 11;
  const int kernel_w = 13;
  vector<int> bottom_shape(4);
  bottom_shape[0] = 15;
  bottom_shape[1] = 18;
  bottom_shape[2] = kernel_h * 2;
  bottom_shape[3] = kernel_w * 2;

  FillerParameter filler_param;
  GaussianFiller<Dtype> filler(filler_param);
  for (int i = 0; i < this->blob_bottom_vec_.size(); ++i) {
    this->blob_bottom_vec_[i]->Reshape(bottom_shape);
    filler.Fill(this->blob_bottom_vec_[i]);
  }

  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->set_num_output(12);
  convolution_param->set_bias_term(false);
  convolution_param->set_group(6);
  convolution_param->add_dilation(2);
  convolution_param->set_kernel_h(kernel_h);
  convolution_param->set_kernel_w(kernel_w);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  Blob<Dtype> weights;
  Blob<Dtype> top_diff;
  // Shape and fill weights and top_diff.
  bool copy_diff;
  bool reshape;
  {
    ConvolutionLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    top_diff.ReshapeLike(*this->blob_top_);
    filler.Fill(&top_diff);
    ASSERT_EQ(1, layer.blobs().size());
    copy_diff = false; reshape = true;
    weights.CopyFrom(*layer.blobs()[0], copy_diff, reshape);
  }

  vector<bool> propagate_down(1, true);
  Blob<Dtype> result_reg;
  Blob<Dtype> backward_result_reg;
  Blob<Dtype> backward_weight_result_reg;
  // Test with regular convolution
  {
    this->blob_bottom_vec_.clear();
    this->blob_top_vec_.clear();
    this->blob_bottom_vec_.push_back(this->blob_bottom_);
    this->blob_top_vec_.push_back(this->blob_top_);
    // Do SetUp and Forward; save Forward result in result_reg.
    convolution_param->set_bottom_is_im2col(false);
    ConvolutionLayer<Dtype> layer_reg(layer_param);
    layer_reg.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    caffe_set(this->blob_top_->count(), Dtype(0),
              this->blob_top_->mutable_cpu_data());
    caffe_set(this->blob_bottom_->count(), Dtype(0),
              this->blob_bottom_->mutable_cpu_diff());
    caffe_set(weights.count(), Dtype(0), weights.mutable_cpu_diff());
    ASSERT_EQ(1, layer_reg.blobs().size());
    copy_diff = false; reshape = false;
    layer_reg.blobs()[0]->CopyFrom(weights, copy_diff, reshape);
    layer_reg.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    copy_diff = false; reshape = true;
    result_reg.CopyFrom(*this->blob_top_, copy_diff, reshape);

    // LOG(ERROR) << "reg blob_bottom_->count() = " << this->blob_bottom_->count();
    // LOG(ERROR) << "reg blob_top_->count() = " << this->blob_top_->count();

    // Copy pre-generated top diff into actual top diff;
    // do Backward and save result in backward_result_reg.
    ASSERT_EQ(this->blob_top_->shape(), top_diff.shape());
    caffe_copy(top_diff.count(), top_diff.cpu_data(),
               this->blob_top_->mutable_cpu_diff());
    layer_reg.Backward(this->blob_top_vec_, propagate_down,
                      this->blob_bottom_vec_);
    copy_diff = true; reshape = true;
    backward_result_reg.CopyFrom(*this->blob_bottom_, copy_diff, reshape);
    //backward_weight_result_reg.CopyFrom(weights, copy_diff, reshape);
    backward_weight_result_reg.CopyFrom(*layer_reg.blobs()[0], copy_diff, reshape);
  }
  Blob<Dtype> result_bii;
  Blob<Dtype> backward_result_bii;
  Blob<Dtype> backward_weight_result_bii;
  // Test with convolution with bottom_is_im2col
  {
    // Do SetUp and Forward; save Forward result in result_bii.
    // do im2col, using blob_bottom/top_
    this->blob_bottom_vec_.clear();
    this->blob_top_vec_.clear();
    this->blob_bottom_vec_.push_back(this->blob_bottom_);
    this->blob_top_vec_.push_back(this->blob_top_);
    Im2colLayer<Dtype> layer_im2col(layer_param);
    layer_im2col.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    caffe_set(this->blob_top_->count(), Dtype(0),
              this->blob_top_->mutable_cpu_data());
    caffe_set(this->blob_bottom_->count(), Dtype(0),
              this->blob_bottom_->mutable_cpu_diff());

    // LOG(ERROR) << "i2c blob_bottom_->count() = " << this->blob_bottom_->count();
    // LOG(ERROR) << "i2c blob_top_->count() = " << this->blob_top_->count();
    
    layer_im2col.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    // for (int i = 0; i < this->blob_top_->count(); ++i) {
    //   LOG(ERROR) << "im2col top[" << i << "] = " << this->blob_top_->cpu_data()[i];
    // }

    // do conv with bottom_is_im2col, using blob_bottom/top_2_
    // reset, and set up blob_bottom/top_2_
    this->blob_bottom_vec_.clear();
    this->blob_top_vec_.clear();
    this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
    this->blob_top_vec_.push_back(this->blob_top_2_);
    // reshape the bottom like the im2col top
    for (int i = 0; i < this->blob_bottom_vec_.size(); ++i) {
      this->blob_bottom_vec_[i]->ReshapeLike(*this->blob_top_);
    }
    convolution_param->set_bottom_is_im2col(true);
    ConvolutionLayer<Dtype> layer_bii(layer_param);
    layer_bii.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

    caffe_set(this->blob_top_2_->count(), Dtype(0),
              this->blob_top_2_->mutable_cpu_data());
    caffe_set(this->blob_bottom_2_->count(), Dtype(0),
              this->blob_bottom_2_->mutable_cpu_diff());
    caffe_set(weights.count(), Dtype(0), weights.mutable_cpu_diff());

    // copy the im2col out into bottom_blob_2_
    caffe_copy(this->blob_top_->count(), this->blob_top_->cpu_data(),
               this->blob_bottom_2_->mutable_cpu_data());
    // LOG(ERROR) << "bii blob_bottom_->count() = " << this->blob_bottom_2_->count();
    // LOG(ERROR) << "bii blob_top_->count() = " << this->blob_top_2_->count();

    ASSERT_EQ(1, layer_bii.blobs().size());    copy_diff = false; reshape = false;
    layer_bii.blobs()[0]->CopyFrom(weights, copy_diff, reshape);
    layer_bii.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    copy_diff = false; reshape = true;
    result_bii.CopyFrom(*this->blob_top_2_, copy_diff, reshape);
    // great! now blob_bottom_top_2_ has a good output, and we've backed it up to result_bii


    // Copy pre-generated top diff into actual top diff;
    // do Backward and save result in backward_result_bii.
    ASSERT_EQ(this->blob_top_2_->shape(), top_diff.shape());
    caffe_copy(top_diff.count(), top_diff.cpu_data(),
               this->blob_top_2_->mutable_cpu_diff());
    // first run the bii conv layer backwards
    layer_bii.Backward(this->blob_top_vec_, propagate_down,
    		       this->blob_bottom_vec_);
    // great! now blob_bottom_blob_2_->cpu_diff() has something for the im2col part
    // copy the bii bottom into the i2c top
    // get blob_bottom/top_ back
    this->blob_bottom_vec_.clear();
    this->blob_top_vec_.clear();
    this->blob_bottom_vec_.push_back(this->blob_bottom_);
    this->blob_top_vec_.push_back(this->blob_top_);
    // reshape
    for (int i = 0; i < this->blob_top_vec_.size(); ++i) {
      this->blob_top_vec_[i]->ReshapeLike(*this->blob_bottom_2_);
    }
    caffe_copy(this->blob_bottom_2_->count(), this->blob_bottom_2_->cpu_diff(),
               this->blob_top_->mutable_cpu_diff());
    layer_im2col.Backward(this->blob_top_vec_, propagate_down,
    			  this->blob_bottom_vec_);
    copy_diff = true; reshape = true;
    backward_result_bii.CopyFrom(*this->blob_bottom_, copy_diff, reshape);
    // backward_weight_result_bii.CopyFrom(weights, copy_diff, reshape);
    backward_weight_result_bii.CopyFrom(*layer_bii.blobs()[0], copy_diff, reshape);
  }
  ASSERT_EQ(result_bii.count(), result_reg.count());
  for (int i = 0; i < result_reg.count(); ++i)  {
    EXPECT_EQ(result_reg.cpu_data()[i], result_bii.cpu_data()[i]);
  }
  ASSERT_EQ(backward_result_bii.count(), backward_result_reg.count());
  for (int i = 0; i < backward_result_reg.count(); ++i) {
    EXPECT_EQ(backward_result_reg.cpu_diff()[i],
              backward_result_bii.cpu_diff()[i]);
  }
  ASSERT_EQ(backward_weight_result_bii.count(),
            backward_weight_result_reg.count());
  for (int i = 0; i < backward_weight_result_reg.count(); ++i) {
    EXPECT_EQ(backward_weight_result_reg.cpu_diff()[i],
              backward_weight_result_bii.cpu_diff()[i]);
  }

  // for (int i = 0; i < result_reg.count(); ++i) {
  //   LOG(ERROR) << "result_reg[" << i << "] = " << result_reg.cpu_data()[i];
  // }
  // for (int i = 0; i < result_bii.count(); ++i) {
  //   LOG(ERROR) << "result_bii[" << i << "] = " << result_bii.cpu_data()[i];
  // }
  // for (int i = 0; i < backward_result_reg.count(); ++i) {
  //   LOG(ERROR) << "backward_result_reg[" << i << "] = " << backward_result_reg.cpu_diff()[i];
  // }
  // for (int i = 0; i < backward_result_bii.count(); ++i) {
  //   LOG(ERROR) << "backward_result_bii[" << i << "] = " << backward_result_bii.cpu_diff()[i];
  // }

  // for (int i = 0; i < backward_weight_result_reg.count(); ++i) {
  //   LOG(ERROR) << "backward_weight_result_reg[" << i << "] = " << backward_weight_result_reg.cpu_diff()[i];
  // }
  // for (int i = 0; i < backward_weight_result_bii.count(); ++i) {
  //   LOG(ERROR) << "backward_weight_result_bii[" << i << "] = " << backward_weight_result_bii.cpu_diff()[i];
  // }


  // for (int i = 0; i < weights.count(); ++i) {
  //   LOG(ERROR) << "weights[" << i << "] = " << weights.cpu_data()[i];
  // }
  // for (int i = 0; i < result_bii.count(); ++i) {
  //   LOG(ERROR) << "result_bii[" << i << "] = " << result_bii.cpu_data()[i];
  // }
}

// TYPED_TEST(ConvolutionLayerTest, TestDilatedGradient) {
//   typedef typename TypeParam::Dtype Dtype;
//   LayerParameter layer_param;
//   ConvolutionParameter* convolution_param =
//       layer_param.mutable_convolution_param();
//   vector<int> bottom_shape;
//   bottom_shape.push_back(2);
//   bottom_shape.push_back(3);
//   bottom_shape.push_back(5);
//   bottom_shape.push_back(6);
//   for (int i = 0; i < this->blob_bottom_vec_.size(); ++i) {
//     this->blob_bottom_vec_[i]->Reshape(bottom_shape);
//   }
//   convolution_param->add_kernel_size(3);
//   convolution_param->add_dilation(2);
//   convolution_param->set_num_output(2);
//   convolution_param->mutable_weight_filler()->set_type("gaussian");
//   convolution_param->mutable_bias_filler()->set_type("gaussian");
//   ConvolutionLayer<Dtype> layer(layer_param);
//   GradientChecker<Dtype> checker(1e-2, 1e-3);
//   checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
//                                   this->blob_top_vec_);
// }

// TYPED_TEST(ConvolutionLayerTest, TestGradient3D) {
//   typedef typename TypeParam::Dtype Dtype;
//   LayerParameter layer_param;
//   ConvolutionParameter* convolution_param =
//       layer_param.mutable_convolution_param();
//   vector<int> bottom_shape(5);
//   bottom_shape[0] = this->blob_bottom_vec_[0]->shape(0);
//   bottom_shape[1] = this->blob_bottom_vec_[0]->shape(1);
//   bottom_shape[2] = 5;
//   bottom_shape[3] = this->blob_bottom_vec_[0]->shape(2);
//   bottom_shape[4] = this->blob_bottom_vec_[0]->shape(3);
//   FillerParameter filler_param;
//   GaussianFiller<Dtype> filler(filler_param);
//   for (int i = 0; i < this->blob_bottom_vec_.size(); ++i) {
//     this->blob_bottom_vec_[i]->Reshape(bottom_shape);
//     filler.Fill(this->blob_bottom_vec_[i]);
//   }
//   convolution_param->add_kernel_size(3);
//   convolution_param->add_stride(2);
//   convolution_param->set_num_output(2);
//   convolution_param->mutable_weight_filler()->set_type("gaussian");
//   convolution_param->mutable_bias_filler()->set_type("gaussian");
//   ConvolutionLayer<Dtype> layer(layer_param);
//   GradientChecker<Dtype> checker(1e-2, 1e-3);
//   checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
//       this->blob_top_vec_);
// }

// TYPED_TEST(ConvolutionLayerTest, Test1x1Gradient) {
//   typedef typename TypeParam::Dtype Dtype;
//   LayerParameter layer_param;
//   ConvolutionParameter* convolution_param =
//       layer_param.mutable_convolution_param();
//   this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
//   this->blob_top_vec_.push_back(this->blob_top_2_);
//   convolution_param->add_kernel_size(1);
//   convolution_param->add_stride(1);
//   convolution_param->set_num_output(2);
//   convolution_param->mutable_weight_filler()->set_type("gaussian");
//   convolution_param->mutable_bias_filler()->set_type("gaussian");
//   ConvolutionLayer<Dtype> layer(layer_param);
//   GradientChecker<Dtype> checker(1e-2, 1e-3);
//   checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
//       this->blob_top_vec_);
// }

// TYPED_TEST(ConvolutionLayerTest, TestGradientGroup) {
//   typedef typename TypeParam::Dtype Dtype;
//   LayerParameter layer_param;
//   ConvolutionParameter* convolution_param =
//       layer_param.mutable_convolution_param();
//   convolution_param->add_kernel_size(3);
//   convolution_param->add_stride(2);
//   convolution_param->set_num_output(3);
//   convolution_param->set_group(3);
//   convolution_param->mutable_weight_filler()->set_type("gaussian");
//   convolution_param->mutable_bias_filler()->set_type("gaussian");
//   ConvolutionLayer<Dtype> layer(layer_param);
//   GradientChecker<Dtype> checker(1e-2, 1e-3);
//   checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
//       this->blob_top_vec_);
// }

// #ifdef USE_CUDNN

// template <typename Dtype>
// class CuDNNConvolutionLayerTest : public GPUDeviceTest<Dtype> {
//  protected:
//   CuDNNConvolutionLayerTest()
//       : blob_bottom_(new Blob<Dtype>(2, 3, 6, 4)),
//         blob_bottom_2_(new Blob<Dtype>(2, 3, 6, 4)),
//         blob_top_(new Blob<Dtype>()),
//         blob_top_2_(new Blob<Dtype>()) {}
//   virtual void SetUp() {
//     // fill the values
//     FillerParameter filler_param;
//     filler_param.set_value(1.);
//     GaussianFiller<Dtype> filler(filler_param);
//     filler.Fill(this->blob_bottom_);
//     filler.Fill(this->blob_bottom_2_);
//     blob_bottom_vec_.push_back(blob_bottom_);
//     blob_top_vec_.push_back(blob_top_);
//   }

//   virtual ~CuDNNConvolutionLayerTest() {
//     delete blob_bottom_;
//     delete blob_bottom_2_;
//     delete blob_top_;
//     delete blob_top_2_;
//   }

//   virtual Blob<Dtype>* MakeReferenceTop(Blob<Dtype>* top) {
//     this->ref_blob_top_.reset(new Blob<Dtype>());
//     this->ref_blob_top_->ReshapeLike(*top);
//     return this->ref_blob_top_.get();
//   }

//   Blob<Dtype>* const blob_bottom_;
//   Blob<Dtype>* const blob_bottom_2_;
//   Blob<Dtype>* const blob_top_;
//   Blob<Dtype>* const blob_top_2_;
//   shared_ptr<Blob<Dtype> > ref_blob_top_;
//   vector<Blob<Dtype>*> blob_bottom_vec_;
//   vector<Blob<Dtype>*> blob_top_vec_;
// };

// TYPED_TEST_CASE(CuDNNConvolutionLayerTest, TestDtypes);

// TYPED_TEST(CuDNNConvolutionLayerTest, TestSetupCuDNN) {
//   this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
//   this->blob_top_vec_.push_back(this->blob_top_2_);
//   LayerParameter layer_param;
//   ConvolutionParameter* convolution_param =
//       layer_param.mutable_convolution_param();
//   convolution_param->add_kernel_size(3);
//   convolution_param->add_stride(2);
//   convolution_param->set_num_output(4);
//   this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
//   this->blob_top_vec_.push_back(this->blob_top_2_);
//   shared_ptr<Layer<TypeParam> > layer(
//       new CuDNNConvolutionLayer<TypeParam>(layer_param));
//   layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
//   EXPECT_EQ(this->blob_top_->num(), 2);
//   EXPECT_EQ(this->blob_top_->channels(), 4);
//   EXPECT_EQ(this->blob_top_->height(), 2);
//   EXPECT_EQ(this->blob_top_->width(), 1);
//   EXPECT_EQ(this->blob_top_2_->num(), 2);
//   EXPECT_EQ(this->blob_top_2_->channels(), 4);
//   EXPECT_EQ(this->blob_top_2_->height(), 2);
//   EXPECT_EQ(this->blob_top_2_->width(), 1);
//   // setting group should not change the shape
//   convolution_param->set_num_output(3);
//   convolution_param->set_group(3);
//   layer.reset(new CuDNNConvolutionLayer<TypeParam>(layer_param));
//   layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
//   EXPECT_EQ(this->blob_top_->num(), 2);
//   EXPECT_EQ(this->blob_top_->channels(), 3);
//   EXPECT_EQ(this->blob_top_->height(), 2);
//   EXPECT_EQ(this->blob_top_->width(), 1);
//   EXPECT_EQ(this->blob_top_2_->num(), 2);
//   EXPECT_EQ(this->blob_top_2_->channels(), 3);
//   EXPECT_EQ(this->blob_top_2_->height(), 2);
//   EXPECT_EQ(this->blob_top_2_->width(), 1);
// }

// TYPED_TEST(CuDNNConvolutionLayerTest, TestSimpleConvolutionCuDNN) {
//   this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
//   this->blob_top_vec_.push_back(this->blob_top_2_);
//   LayerParameter layer_param;
//   ConvolutionParameter* convolution_param =
//       layer_param.mutable_convolution_param();
//   convolution_param->add_kernel_size(3);
//   convolution_param->add_stride(2);
//   convolution_param->set_num_output(4);
//   convolution_param->mutable_weight_filler()->set_type("gaussian");
//   convolution_param->mutable_bias_filler()->set_type("constant");
//   convolution_param->mutable_bias_filler()->set_value(0.1);
//   shared_ptr<Layer<TypeParam> > layer(
//       new CuDNNConvolutionLayer<TypeParam>(layer_param));
//   layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
//   layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
//   // Check against reference convolution.
//   const TypeParam* top_data;
//   const TypeParam* ref_top_data;
//   caffe_conv(this->blob_bottom_, convolution_param, layer->blobs(),
//       this->MakeReferenceTop(this->blob_top_));
//   top_data = this->blob_top_->cpu_data();
//   ref_top_data = this->ref_blob_top_->cpu_data();
//   for (int i = 0; i < this->blob_top_->count(); ++i) {
//     EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
//   }
//   caffe_conv(this->blob_bottom_2_, convolution_param, layer->blobs(),
//       this->MakeReferenceTop(this->blob_top_2_));
//   top_data = this->blob_top_2_->cpu_data();
//   ref_top_data = this->ref_blob_top_->cpu_data();
//   for (int i = 0; i < this->blob_top_->count(); ++i) {
//     EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
//   }
// }

// TYPED_TEST(CuDNNConvolutionLayerTest, TestSimpleConvolutionGroupCuDNN) {
//   LayerParameter layer_param;
//   ConvolutionParameter* convolution_param =
//       layer_param.mutable_convolution_param();
//   convolution_param->add_kernel_size(3);
//   convolution_param->add_stride(2);
//   convolution_param->set_num_output(3);
//   convolution_param->set_group(3);
//   convolution_param->mutable_weight_filler()->set_type("gaussian");
//   convolution_param->mutable_bias_filler()->set_type("constant");
//   convolution_param->mutable_bias_filler()->set_value(0.1);
//   shared_ptr<Layer<TypeParam> > layer(
//       new CuDNNConvolutionLayer<TypeParam>(layer_param));
//   layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
//   layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
//   // Check against reference convolution.
//   const TypeParam* top_data;
//   const TypeParam* ref_top_data;
//   caffe_conv(this->blob_bottom_, convolution_param, layer->blobs(),
//       this->MakeReferenceTop(this->blob_top_));
//   top_data = this->blob_top_->cpu_data();
//   ref_top_data = this->ref_blob_top_->cpu_data();
//   for (int i = 0; i < this->blob_top_->count(); ++i) {
//     EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
//   }
// }

// TYPED_TEST(CuDNNConvolutionLayerTest, TestSobelConvolutionCuDNN) {
//   // Test separable convolution by computing the Sobel operator
//   // as a single filter then comparing the result
//   // as the convolution of two rectangular filters.

//   // Fill bottoms with identical Gaussian noise.
//   shared_ptr<GaussianFiller<TypeParam> > filler;
//   FillerParameter filler_param;
//   filler_param.set_value(1.);
//   filler.reset(new GaussianFiller<TypeParam>(filler_param));
//   filler->Fill(this->blob_bottom_);
//   this->blob_bottom_2_->CopyFrom(*this->blob_bottom_);
//   // Compute Sobel G_x operator as 3 x 3 convolution.
//   LayerParameter layer_param;
//   ConvolutionParameter* convolution_param =
//       layer_param.mutable_convolution_param();
//   convolution_param->add_kernel_size(3);
//   convolution_param->add_stride(2);
//   convolution_param->set_num_output(1);
//   convolution_param->set_bias_term(false);
//   shared_ptr<Layer<TypeParam> > layer(
//       new CuDNNConvolutionLayer<TypeParam>(layer_param));
//   layer->blobs().resize(1);
//   layer->blobs()[0].reset(new Blob<TypeParam>(1, 3, 3, 3));
//   TypeParam* weights = layer->blobs()[0]->mutable_cpu_data();
//   for (int c = 0; c < 3; ++c) {
//     int i = c * 9;  // 3 x 3 filter
//     weights[i +  0] = -1;
//     weights[i +  1] =  0;
//     weights[i +  2] =  1;
//     weights[i +  3] = -2;
//     weights[i +  4] =  0;
//     weights[i +  5] =  2;
//     weights[i +  6] = -1;
//     weights[i +  7] =  0;
//     weights[i +  8] =  1;
//   }
//   layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
//   layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
//   // Compute Sobel G_x operator as separable 3 x 1 and 1 x 3 convolutions.
//   // (1) the [1 2 1] column filter
//   vector<Blob<TypeParam>*> sep_blob_bottom_vec;
//   vector<Blob<TypeParam>*> sep_blob_top_vec;
//   shared_ptr<Blob<TypeParam> > blob_sep(new Blob<TypeParam>());
//   sep_blob_bottom_vec.push_back(this->blob_bottom_2_);
//   sep_blob_top_vec.push_back(this->blob_top_2_);
//   convolution_param->clear_kernel_size();
//   convolution_param->clear_stride();
//   convolution_param->set_kernel_h(3);
//   convolution_param->set_kernel_w(1);
//   convolution_param->set_stride_h(2);
//   convolution_param->set_stride_w(1);
//   convolution_param->set_num_output(1);
//   convolution_param->set_bias_term(false);
//   layer.reset(new CuDNNConvolutionLayer<TypeParam>(layer_param));
//   layer->blobs().resize(1);
//   layer->blobs()[0].reset(new Blob<TypeParam>(1, 3, 3, 1));
//   TypeParam* weights_1 = layer->blobs()[0]->mutable_cpu_data();
//   for (int c = 0; c < 3; ++c) {
//     int i = c * 3;  // 3 x 1 filter
//     weights_1[i +  0] = 1;
//     weights_1[i +  1] = 2;
//     weights_1[i +  2] = 1;
//   }
//   layer->SetUp(sep_blob_bottom_vec, sep_blob_top_vec);
//   layer->Forward(sep_blob_bottom_vec, sep_blob_top_vec);
//   // (2) the [-1 0 1] row filter
//   blob_sep->CopyFrom(*this->blob_top_2_, false, true);
//   sep_blob_bottom_vec.clear();
//   sep_blob_bottom_vec.push_back(blob_sep.get());
//   convolution_param->set_kernel_h(1);
//   convolution_param->set_kernel_w(3);
//   convolution_param->set_stride_h(1);
//   convolution_param->set_stride_w(2);
//   convolution_param->set_num_output(1);
//   convolution_param->set_bias_term(false);
//   layer.reset(new CuDNNConvolutionLayer<TypeParam>(layer_param));
//   layer->blobs().resize(1);
//   layer->blobs()[0].reset(new Blob<TypeParam>(1, 1, 1, 3));
//   TypeParam* weights_2 = layer->blobs()[0]->mutable_cpu_data();
//   weights_2[0] = -1;
//   weights_2[1] =  0;
//   weights_2[2] =  1;
//   layer->SetUp(sep_blob_bottom_vec, sep_blob_top_vec);
//   layer->Forward(sep_blob_bottom_vec, sep_blob_top_vec);
//   // Test equivalence of full and separable filters.
//   const TypeParam* top_data = this->blob_top_->cpu_data();
//   const TypeParam* sep_top_data = this->blob_top_2_->cpu_data();
//   for (int i = 0; i < this->blob_top_->count(); ++i) {
//     EXPECT_NEAR(top_data[i], sep_top_data[i], 1e-4);
//   }
// }

// TYPED_TEST(CuDNNConvolutionLayerTest, TestGradientCuDNN) {
//   LayerParameter layer_param;
//   ConvolutionParameter* convolution_param =
//       layer_param.mutable_convolution_param();
//   this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
//   this->blob_top_vec_.push_back(this->blob_top_2_);
//   convolution_param->add_kernel_size(3);
//   convolution_param->add_stride(2);
//   convolution_param->set_num_output(2);
//   convolution_param->mutable_weight_filler()->set_type("gaussian");
//   convolution_param->mutable_bias_filler()->set_type("gaussian");
//   CuDNNConvolutionLayer<TypeParam> layer(layer_param);
//   GradientChecker<TypeParam> checker(1e-2, 1e-3);
//   checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
//       this->blob_top_vec_);
// }

// TYPED_TEST(CuDNNConvolutionLayerTest, TestGradientGroupCuDNN) {
//   LayerParameter layer_param;
//   ConvolutionParameter* convolution_param =
//       layer_param.mutable_convolution_param();
//   convolution_param->add_kernel_size(3);
//   convolution_param->add_stride(2);
//   convolution_param->set_num_output(3);
//   convolution_param->set_group(3);
//   convolution_param->mutable_weight_filler()->set_type("gaussian");
//   convolution_param->mutable_bias_filler()->set_type("gaussian");
//   CuDNNConvolutionLayer<TypeParam> layer(layer_param);
//   GradientChecker<TypeParam> checker(1e-2, 1e-3);
//   checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
//       this->blob_top_vec_);
// }

// #endif

}  // namespace caffe

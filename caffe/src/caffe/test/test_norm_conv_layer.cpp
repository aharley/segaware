#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/norm_conv_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class NormConvLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

protected:
  // NormConvLayerTest()
  //     : blob_bottom_(new Blob<Dtype>(3, 3, 7, 5)),
  //       blob_bottom_2_(new Blob<Dtype>(3, 10, 7, 5)),
  //       blob_top_(new Blob<Dtype>()) {}
  NormConvLayerTest()
    : blob_bottom_(new Blob<Dtype>(2, 1, 3, 4)),
      blob_bottom_2_(new Blob<Dtype>(2, 2, 3, 4)),
      blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_value(1.);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    filler.Fill(this->blob_bottom_2_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_2_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~NormConvLayerTest() {
    delete blob_bottom_;
    delete blob_bottom_2_;
    delete blob_top_;
  }

  // virtual Blob<Dtype>* MakeReferenceTop(Blob<Dtype>* top) {
  //   this->ref_blob_top_.reset(new Blob<Dtype>());
  //   this->ref_blob_top_->ReshapeLike(*top);
  //   return this->ref_blob_top_.get();
  // }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_2_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(NormConvLayerTest, TestDtypesAndDevices);

// TYPED_TEST(NormConvLayerTest, TestSetup) {
//   typedef typename TypeParam::Dtype Dtype;
//   LayerParameter layer_param;
//   ConvolutionParameter* convolution_param =
//       layer_param.mutable_convolution_param();
//   convolution_param->add_kernel_size(3);
//   convolution_param->add_stride(2);
//   convolution_param->set_num_output(4);
//   // convolution_param->add_kernel_size(3);
//   // convolution_param->add_stride(1);
//   // convolution_param->set_num_output(1);
//   convolution_param->set_normalized_convolution(true);
//   shared_ptr<Layer<Dtype> > layer(
//       new NormConvLayer<Dtype>(layer_param));
//   layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
//   EXPECT_EQ(this->blob_top_->num(), 2);
//   EXPECT_EQ(this->blob_top_->channels(), 4);
//   EXPECT_EQ(this->blob_top_->height(), 2);
//   EXPECT_EQ(this->blob_top_->width(), 1);
//   // EXPECT_EQ(this->blob_top_->num(), 2);
//   // EXPECT_EQ(this->blob_top_->channels(), 1);
//   // EXPECT_EQ(this->blob_top_->height(), 1);
//   // EXPECT_EQ(this->blob_top_->width(), 1);
//   // // setting group should not change the shape
//   // convolution_param->set_num_output(3);
//   // convolution_param->set_group(3);
//   // layer.reset(new NormConvLayer<Dtype>(layer_param));
//   // layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
//   // EXPECT_EQ(this->blob_top_->num(), 2);
//   // EXPECT_EQ(this->blob_top_->channels(), 3);
//   // EXPECT_EQ(this->blob_top_->height(), 2);
//   // EXPECT_EQ(this->blob_top_->width(), 1);
// }

// TYPED_TEST(NormConvLayerTest, TestSimpleNormConv) {
//   typedef typename TypeParam::Dtype Dtype;
//   this->blob_bottom_vec_.clear();
//   this->blob_top_vec_.clear();
//   this->blob_bottom_vec_.push_back(this->blob_bottom_); // img
//   this->blob_bottom_vec_.push_back(this->blob_bottom_2_); // emb
//   this->blob_top_vec_.push_back(this->blob_top_);
//   LayerParameter layer_param;
//   ConvolutionParameter* convolution_param =
//       layer_param.mutable_convolution_param();
//   NormConvParameter* normalized_convolution_param =
//       layer_param.mutable_normalized_convolution_param();
//   convolution_param->add_kernel_size(3);
//   convolution_param->add_stride(1);
//   convolution_param->add_pad(0);
//   convolution_param->set_num_output(1);
//   convolution_param->mutable_weight_filler()->set_type("gaussian");
//   // convolution_param->mutable_bias_filler()->set_type("constant");
//   // convolution_param->mutable_bias_filler()->set_value(0.1);
//   convolution_param->set_bias_term(false);
//   convolution_param->set_normalized_convolution(true);
//   normalized_convolution_param->set_scale_term(false);

//   int img_channels = 1;
//   int emb_channels = 1;

//   vector<int> bottom_shape;
//   bottom_shape.push_back(1);
//   bottom_shape.push_back(img_channels);
//   bottom_shape.push_back(3);
//   bottom_shape.push_back(5);
//   vector<int> emb_shape;
//   emb_shape.push_back(1);
//   emb_shape.push_back(emb_channels);
//   emb_shape.push_back(3);
//   emb_shape.push_back(5);

//   this->blob_bottom_vec_[0]->Reshape(bottom_shape);
//   this->blob_bottom_vec_[1]->Reshape(emb_shape);

//   // FillerParameter filler_param;
//   // filler_param.set_value(1.);
//   // GaussianFiller<Dtype> filler(filler_param);
//   // for (int i = 0; i < this->blob_bottom_vec_.size(); ++i) {
//   //   this->blob_bottom_vec_[i]->Reshape(bottom_shape);
//   //   //filler.Fill(this->blob_bottom_vec_[i]);
//   // }
//   for (int i = 0; i < 15 * 1 * img_channels; i += 15) {
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
//   for (int i = 0; i < 15 * 1 * emb_channels; i += 15) {
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

//   // FillerParameter filler_param;
//   // GaussianFiller<Dtype> filler(filler_param);

//   // normalized_convolution_param->set_normalized_convolution(true);
//   normalized_convolution_param->set_norm(NormConvParameter_Norm_L1);

//   shared_ptr<Layer<Dtype> > layer(
//       new NormConvLayer<Dtype>(layer_param));
//   layer->blobs().resize(1);
//   layer->blobs()[0].reset(new Blob<Dtype>(1, img_channels, 3, 3)); // filter shape
//   Dtype* weights = layer->blobs()[0]->mutable_cpu_data();
//   for (int c = 0; c < img_channels; ++c) {
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
//   // convolution_param->mutable_bias_filler()->set_type("constant");
//   // convolution_param->mutable_bias_filler()->set_value(0.1);
//   // vector<int> bias_shape(1, 1);
//   // layer->blobs()[1].reset(new Blob<Dtype>(bias_shape));
//   // layer->blobs()[1]->mutable_cpu_data()[0] = 0.5;

//   // normalized_convolution_param->mutable_scale_filler()->set_type("constant");
//   // normalized_convolution_param->mutable_scale_filler()->set_value(0.1);
//   // vector<int> scale_shape(1, 1);
//   // layer->blobs()[2].reset(new Blob<Dtype>(scale_shape));
//   // layer->blobs()[2]->mutable_cpu_data()[0] = 0.5;

//   layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
//   layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);

//   const Dtype* top_data = this->blob_top_->cpu_data();
//   for (int i=0; i < this->blob_top_->count(); i++) {
//     LOG(ERROR) << "top[" << i << "] = " << top_data[i];
//   }
// }

TYPED_TEST(NormConvLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  NormConvParameter* norm_conv_param =
      layer_param.mutable_norm_conv_param();
  convolution_param->add_kernel_size(3);
  convolution_param->add_stride(1);
  convolution_param->set_num_output(1);
  convolution_param->add_dilation(1);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  // convolution_param->set_bias_term(false);
  convolution_param->set_bias_term(true);
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  // convolution_param->mutable_bias_filler()->set_type("constant");
  // convolution_param->mutable_bias_filler()->set_value(0.5);

  norm_conv_param->set_norm(NormConvParameter_Norm_L2);
  norm_conv_param->mutable_scale_filler()->set_value(0.1);
  // norm_conv_param->mutable_scale_filler()->set_type("gaussian");
  // norm_conv_param->set_scale_term(false);

  // int num = 1;
  // int img_channels = 1;
  // int emb_channels = 1;
  // int height = 3;
  // int width = 3;
  // vector<int> bottom1_shape;
  // bottom1_shape.push_back(num);
  // bottom1_shape.push_back(img_channels);
  // bottom1_shape.push_back(height);
  // bottom1_shape.push_back(width);
  // vector<int> bottom2_shape;
  // bottom2_shape.push_back(num);
  // bottom2_shape.push_back(emb_channels);
  // bottom2_shape.push_back(height);
  // bottom2_shape.push_back(width);
  // this->blob_bottom_vec_[0]->Reshape(bottom1_shape);
  // this->blob_bottom_vec_[1]->Reshape(bottom2_shape);

  // for (int i = 0; i < 9 * num * img_channels; i += 9) {
  //   this->blob_bottom_->mutable_cpu_data()[i +  0] = 10;
  //   this->blob_bottom_->mutable_cpu_data()[i +  1] = 2;
  //   this->blob_bottom_->mutable_cpu_data()[i +  2] = 5;
  //   this->blob_bottom_->mutable_cpu_data()[i +  3] = 2;
  //   this->blob_bottom_->mutable_cpu_data()[i +  4] = 3;
  //   this->blob_bottom_->mutable_cpu_data()[i +  5] = 9;
  //   this->blob_bottom_->mutable_cpu_data()[i +  6] = 1;
  //   this->blob_bottom_->mutable_cpu_data()[i +  7] = 1 + i;
  //   this->blob_bottom_->mutable_cpu_data()[i +  8] = 1 + i;
  //   // this->blob_bottom_->mutable_cpu_data()[i +  9] = 8;
  //   // this->blob_bottom_->mutable_cpu_data()[i + 10] = 1;
  //   // this->blob_bottom_->mutable_cpu_data()[i + 11] = 2;
  //   // this->blob_bottom_->mutable_cpu_data()[i + 12] = 5;
  //   // this->blob_bottom_->mutable_cpu_data()[i + 13] = 2;
  //   // this->blob_bottom_->mutable_cpu_data()[i + 14] = 3;
  // }
  // for (int i = 0; i < 9 * num * emb_channels; i += 9) {
  //   this->blob_bottom_2_->mutable_cpu_data()[i +  0] = 15;
  //   this->blob_bottom_2_->mutable_cpu_data()[i +  1] = 20;
  //   this->blob_bottom_2_->mutable_cpu_data()[i +  2] = 50;
  //   this->blob_bottom_2_->mutable_cpu_data()[i +  3] = 2;
  //   this->blob_bottom_2_->mutable_cpu_data()[i +  4] = 3;
  //   this->blob_bottom_2_->mutable_cpu_data()[i +  5] = 9;
  //   this->blob_bottom_2_->mutable_cpu_data()[i +  6] = 1;
  //   this->blob_bottom_2_->mutable_cpu_data()[i +  7] = 9;
  //   this->blob_bottom_2_->mutable_cpu_data()[i +  8] = 10;
  //   // this->blob_bottom_2_->mutable_cpu_data()[i +  9] = 8;
  //   // this->blob_bottom_2_->mutable_cpu_data()[i + 10] = 1;
  //   // this->blob_bottom_2_->mutable_cpu_data()[i + 11] = 2;
  //   // this->blob_bottom_2_->mutable_cpu_data()[i + 12] = 5;
  //   // this->blob_bottom_2_->mutable_cpu_data()[i + 13] = 2;
  //   // this->blob_bottom_2_->mutable_cpu_data()[i + 14] = 3;
  // }
  // for (int i = 0; i < height*width*num*img_channels; i++) {
  //   this->blob_bottom_->mutable_cpu_data()[i] = i*i;
  // }
  // for (int i = 0; i < height*width*num*emb_channels; i++) {
  //   this->blob_bottom_2_->mutable_cpu_data()[i] = i*i;
  // }

  // LOG(ERROR) << "ok got it";
  // const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  // for (int i=0; i < this->blob_bottom_->count(); i++) {
  //   LOG(ERROR) << "img_bottom[" << i << "] = " << bottom_data[i];
  // }
  // const Dtype* bottom2_data = this->blob_bottom_2_->cpu_data();
  // for (int i=0; i < this->blob_bottom_2_->count(); i++) {
  //   LOG(ERROR) << "emb_bottom[" << i << "] = " << bottom2_data[i];
  // }

  NormConvLayer<Dtype> layer(layer_param);

  // layer.blobs().resize(2);
  // layer.blobs()[0].reset(new Blob<Dtype>(1, img_channels, 3, 3));
  // Dtype* weights = layer.blobs()[0]->mutable_cpu_data();
  // for (int c = 0; c < img_channels; ++c) {
  //   int i = c * 9;  // 3 x 3 filter
  //   weights[i +  0] = -2;
  //   weights[i +  1] =  0;
  //   weights[i +  2] =  1;
  //   weights[i +  3] = -2;
  //   weights[i +  4] =  0;
  //   weights[i +  5] =  2;
  //   weights[i +  6] = -1;
  //   weights[i +  7] =  0;
  //   weights[i +  8] =  1;
  // }
  // vector<int> bias_shape(1, 1);
  // layer.blobs()[1].reset(new Blob<Dtype>(bias_shape));
  // layer.blobs()[1]->mutable_cpu_data()[0] = 0.5;
  // vector<int> scale_shape(1, 1);
  // layer.blobs()[2].reset(new Blob<Dtype>(scale_shape));
  // layer.blobs()[2]->mutable_cpu_data()[0] = 0.01;

  // layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  // layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  
  // const Dtype* top_data = this->blob_top_->cpu_data();
  // for (int i=0; i < this->blob_top_->count(); i++) {
  //   LOG(ERROR) << "top[" << i << "] = " << top_data[i];
  // }
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}




}  // namespace caffe

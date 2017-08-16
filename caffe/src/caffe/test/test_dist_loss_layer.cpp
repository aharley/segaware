#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/dist_loss_layer.hpp"
#include "caffe/layers/im2dist_layer.hpp"
#include "caffe/layers/im2parity_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class DistLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  DistLossLayerTest()
    : im2dist_blob_bottom_data_(new Blob<Dtype>(1, 1, 3, 5)),
      im2dist_blob_top_data_(new Blob<Dtype>(1, 9, 3, 5)),
      im2parity_blob_bottom_data_(new Blob<Dtype>(1, 1, 3, 5)),
      im2parity_blob_top_data_(new Blob<Dtype>(1, 9, 3, 5)),
      blob_top_loss_(new Blob<Dtype>()) {
    // blob_bottom_data_(new Blob<Dtype>(1, 9, 3, 5)),
    // blob_bottom_label_(new Blob<Dtype>(1, 9, 3, 5)),

    // // fill the values
    // FillerParameter filler_param;
    // //filler_param.set_std(10);
    // GaussianFiller<Dtype> filler(filler_param);
    // filler.Fill(this->blob_bottom_data_);
    // blob_bottom_vec_.push_back(blob_bottom_data_);
    // for (int i = 0; i < blob_bottom_label_->count(); ++i) {
    //   blob_bottom_label_->mutable_cpu_data()[i] = caffe_rng_rand() % 3;
    // }

    im2dist_blob_bottom_vec_.push_back(im2dist_blob_bottom_data_);
    im2dist_blob_top_vec_.push_back(im2dist_blob_top_data_);
    //im2dist_blob_top_vec_.push_back(blob_bottom_data_);

    im2parity_blob_bottom_vec_.push_back(im2parity_blob_bottom_data_);
    im2parity_blob_top_vec_.push_back(im2parity_blob_top_data_);
    //im2parity_blob_top_vec_.push_back(blob_bottom_label_);

    blob_bottom_vec_.push_back(im2dist_blob_top_data_);
    blob_bottom_vec_.push_back(im2parity_blob_top_data_);
    blob_top_vec_.push_back(blob_top_loss_);
    // int num=1;
    // int channels=1;
    // for (int i = 0; i < 3*5 * num * channels; i += 15) {
    //   blob_bottom_data_->mutable_cpu_data()[i +  0] = 1;
    //   blob_bottom_data_->mutable_cpu_data()[i +  1] = 2;
    //   blob_bottom_data_->mutable_cpu_data()[i +  2] = 5;
    //   blob_bottom_data_->mutable_cpu_data()[i +  3] = 5;
    //   blob_bottom_data_->mutable_cpu_data()[i +  4] = 5;
    //   blob_bottom_data_->mutable_cpu_data()[i +  5] = 9;
    //   blob_bottom_data_->mutable_cpu_data()[i +  6] = 1;
    //   blob_bottom_data_->mutable_cpu_data()[i +  7] = 1;
    //   blob_bottom_data_->mutable_cpu_data()[i +  8] = 4;
    //   blob_bottom_data_->mutable_cpu_data()[i +  9] = 8;
    //   blob_bottom_data_->mutable_cpu_data()[i + 10] = 1;
    //   blob_bottom_data_->mutable_cpu_data()[i + 11] = 2;
    //   blob_bottom_data_->mutable_cpu_data()[i + 12] = 5;
    //   blob_bottom_data_->mutable_cpu_data()[i + 13] = 2;
    //   blob_bottom_data_->mutable_cpu_data()[i + 14] = 3;
    //   //blob_bottom_data_->mutable_cpu_data()[i + 15] = 2;
    // }
    
    // for (int i=0; i<3*5; i++){
    //   blob_bottom_label_->mutable_cpu_data()[i] = 0;
    // }
    // blob_bottom_label_->mutable_cpu_data()[0] = 1;
    // blob_bottom_label_->mutable_cpu_data()[1] = 1;
    // blob_bottom_label_->mutable_cpu_data()[6] = 1;
    // blob_bottom_label_->mutable_cpu_data()[7] = 1;
    // blob_bottom_label_->mutable_cpu_data()[8] = 1;

    // for (int i=0; i < 3*5*num*channels; i++){
    //   LOG(ERROR) << "data[" << i << "]=" << blob_bottom_data_->mutable_cpu_data()[i];
    // }
    // for (int i=0; i < 3*5*num; i++){
    //   LOG(ERROR) << "label[" << i << "]=" << blob_bottom_label_->mutable_cpu_data()[i];
    // }
    
    // blob_bottom_vec_.push_back(blob_bottom_label_);
    // blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~DistLossLayerTest() {
    delete im2dist_blob_bottom_data_;
    delete im2dist_blob_top_data_;
    delete im2parity_blob_bottom_data_;
    delete im2parity_blob_top_data_;
    // delete im2parity_blob_bottom_data_;
    //delete blob_bottom_data_;
    //delete blob_bottom_label_;
    delete blob_top_loss_;
  }
  Blob<Dtype>* const im2dist_blob_bottom_data_;
  Blob<Dtype>* const im2dist_blob_top_data_;
  vector<Blob<Dtype>*> im2dist_blob_bottom_vec_;
  vector<Blob<Dtype>*> im2dist_blob_top_vec_;
  Blob<Dtype>* const im2parity_blob_bottom_data_;
  Blob<Dtype>* const im2parity_blob_top_data_;
  vector<Blob<Dtype>*> im2parity_blob_bottom_vec_;
  vector<Blob<Dtype>*> im2parity_blob_top_vec_;
  // Blob<Dtype>* const im2parity_blob_bottom_data_;
  // Blob<Dtype>* const blob_bottom_data_;
  // Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_top_loss_;
  // vector<Blob<Dtype>*> im2parity_blob_bottom_vec_;
  // vector<Blob<Dtype>*> im2parity_blob_top_vec_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;

  void TestForwardSquare() {
    int stride = 1;
    int kernel = 3;
    int pad = 1;
    LayerParameter shared_param;
    ConvolutionParameter* conv_param = shared_param.mutable_convolution_param();
    conv_param->add_kernel_size(kernel);
    conv_param->add_stride(stride);
    conv_param->add_pad(pad);

    const int num = 1;
    const int channels = 1;
    const int height = 3; 
    const int width = 3;
    
    // Image
    // [1 2 5]
    // [2 3 9]
    // [4 1 4]
    im2dist_blob_bottom_data_->Reshape(num, channels, height, width);
    for (int i = 0; i < height*width * num * channels; i += height*width) {
      im2dist_blob_bottom_data_->mutable_cpu_data()[i +  0] = 1;
      im2dist_blob_bottom_data_->mutable_cpu_data()[i +  1] = 2;
      im2dist_blob_bottom_data_->mutable_cpu_data()[i +  2] = 5;
      im2dist_blob_bottom_data_->mutable_cpu_data()[i +  3] = 2;
      im2dist_blob_bottom_data_->mutable_cpu_data()[i +  4] = 3;
      im2dist_blob_bottom_data_->mutable_cpu_data()[i +  5] = 9;
      im2dist_blob_bottom_data_->mutable_cpu_data()[i +  6] = 4;
      im2dist_blob_bottom_data_->mutable_cpu_data()[i +  7] = 1;
      im2dist_blob_bottom_data_->mutable_cpu_data()[i +  8] = 4;
    }
    Im2distLayer<Dtype> im2dist_layer(shared_param);
    im2dist_layer.SetUp(im2dist_blob_bottom_vec_, im2dist_blob_top_vec_);
    im2dist_layer.Forward(im2dist_blob_bottom_vec_, im2dist_blob_top_vec_);
    // for (int i = 0; i < im2dist_blob_top_data_->count(); ++i) {
    //   LOG(ERROR) << "im2dist_blob_top_data[" << i << "] = " 
    // 		 << im2dist_blob_top_data_->mutable_cpu_data()[i];
    // }
    
    // Labels
    // [1 2 3]
    // [4 5 6]
    // [1 1 9]
    im2parity_blob_bottom_data_->Reshape(num, channels, height, width);
    for (int i=0; i<height*width; i++) {
       im2parity_blob_bottom_data_->mutable_cpu_data()[i] = i+1;
    }
    im2parity_blob_bottom_data_->mutable_cpu_data()[6] = 1;
    im2parity_blob_bottom_data_->mutable_cpu_data()[7] = 1;
    Im2parityLayer<Dtype> im2parity_layer(shared_param);
    im2parity_layer.SetUp(im2parity_blob_bottom_vec_, im2parity_blob_top_vec_);
    im2parity_layer.Forward(im2parity_blob_bottom_vec_, im2parity_blob_top_vec_);
    // for (int i = 0; i < im2parity_blob_top_data_->count(); ++i) {
    //   LOG(ERROR) << "im2parity_blob_top_data[" << i << "] = " << im2parity_blob_top_data_->mutable_cpu_data()[i];
    // }

    LayerParameter layer_param;
    DistLossParameter* dist_loss_param = layer_param.mutable_dist_loss_param();
    dist_loss_param->set_alpha(0.5);
    dist_loss_param->set_beta(2);

    ConvolutionParameter* conv_param2 = layer_param.mutable_convolution_param();
    conv_param2->add_kernel_size(kernel);
    conv_param2->add_stride(stride);
    conv_param2->add_pad(pad);

    DistLossLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    layer.Forward(blob_bottom_vec_, blob_top_vec_);

    const Dtype loss = blob_top_loss_->cpu_data()[0];
    // LOG(ERROR) << "loss = " << loss;
    // LOG(ERROR) << "count = " << im2dist_blob_top_data_->count();

    // the value 45 below was computed with a matlab implementation.
    LOG(ERROR) << "loss by layer is " << blob_top_loss_->cpu_data()[0];
    LOG(ERROR) << "bottom count is " << im2dist_blob_bottom_data_->count();
    LOG(ERROR) << "top count is " << im2dist_blob_bottom_data_->count();
    EXPECT_NEAR(blob_top_loss_->cpu_data()[0], Dtype(45)/im2dist_blob_top_data_->count(), 0.01);
    EXPECT_EQ(blob_top_loss_->num(), 1);
    EXPECT_EQ(blob_top_loss_->channels(), 1);
    EXPECT_EQ(blob_top_loss_->height(), 1);
    EXPECT_EQ(blob_top_loss_->width(), 1);
  }
  
  // void TestBackwardSquare() {
  //   LOG(ERROR) << "testing backward square";
  //   LayerParameter layer_param;
  //   DistLossParameter* dist_loss_param = layer_param.mutable_dist_loss_param();
  //   dist_loss_param->set_kernel_size(3);
  //   dist_loss_param->set_stride(3);
  //   dist_loss_param->set_num_output(5);
  //   //DistLossLayer<Dtype> layer(layer_param);

  //   const int num = 1;
  //   const int channels = 4;
  //   blob_bottom_data_->Reshape(num, channels, 3, 5);
  //   // Input: 1x 2 channels of:
  //   //     [1 2 5 2 3]
  //   //     [9 4 1 4 8]
  //   //     [1 2 5 2 3]
  //   for (int i = 0; i < 15 * num * channels; i += 15) {
  //     blob_bottom_data_->mutable_cpu_data()[i +  0] = 1;
  //     blob_bottom_data_->mutable_cpu_data()[i +  1] = 2;
  //     blob_bottom_data_->mutable_cpu_data()[i +  2] = 5;
  //     blob_bottom_data_->mutable_cpu_data()[i +  3] = 2;
  //     blob_bottom_data_->mutable_cpu_data()[i +  4] = 3;
  //     blob_bottom_data_->mutable_cpu_data()[i +  5] = 9;
  //     blob_bottom_data_->mutable_cpu_data()[i +  6] = 4;
  //     blob_bottom_data_->mutable_cpu_data()[i +  7] = 1;
  //     blob_bottom_data_->mutable_cpu_data()[i +  8] = 4;
  //     blob_bottom_data_->mutable_cpu_data()[i +  9] = 8;
  //     blob_bottom_data_->mutable_cpu_data()[i + 10] = 1;
  //     blob_bottom_data_->mutable_cpu_data()[i + 11] = 2;
  //     blob_bottom_data_->mutable_cpu_data()[i + 12] = 5;
  //     blob_bottom_data_->mutable_cpu_data()[i + 13] = 2;
  //     blob_bottom_data_->mutable_cpu_data()[i + 14] = 3;
  //   }
  //   // Labels: 1x 1 channels of:
  //   //     [0 0 0 0 0]
  //   //     [0 1 1 0 0]
  //   //     [0 0 0 0 0]
  //   for (int i=0; i<15; i++){
  //     blob_bottom_label_->mutable_cpu_data()[i] = 0;
  //   }
  //   blob_bottom_label_->mutable_cpu_data()[0] = 1;
  //   blob_bottom_label_->mutable_cpu_data()[6] = 1;
  //   blob_bottom_label_->mutable_cpu_data()[7] = 1;

  //   LOG(ERROR) << "set up input and label";

  //   DistLossLayer<Dtype> layer(layer_param);
  //   layer.SetUp(blob_bottom_vec_, blob_top_vec_);
  //   EXPECT_EQ(blob_top_loss_->num(), 1);
  //   EXPECT_EQ(blob_top_loss_->channels(), 1);
  //   EXPECT_EQ(blob_top_loss_->height(), 1);
  //   EXPECT_EQ(blob_top_loss_->width(), 1);
  //   layer.Forward(blob_bottom_vec_, blob_top_vec_);
  //   vector<bool> propagate_down(1, false);
  //   propagate_down[0] = true;
  //   layer.Backward(blob_top_vec_, propagate_down, blob_bottom_vec_);
  //   for (int i=0; i<15; i++){
  //     LOG(ERROR) << "at i=" << i << ", bottomdata_diff = " << blob_bottom_data_->mutable_cpu_diff()[i];
  //   }
  //   // for (int i=0; i<15; i++){
  //   //   LOG(ERROR) << "at i=" << i << ", top_loss_diff = " << blob_top_loss_->mutable_cpu_diff()[i];
  //   // }
  //   // Expected output: with stride=2, ans =-21.5; with stride=2, ans=-3.5
  //   //EXPECT_EQ(blob_top_loss_->cpu_data()[0], -3.5);
  // }
};

TYPED_TEST_CASE(DistLossLayerTest, TestDtypesAndDevices);

// TYPED_TEST(DistLossLayerTest, TestSetup) {
//   typedef typename TypeParam::Dtype Dtype;
//   LayerParameter layer_param;
//   DistLossParameter* dist_loss_param = layer_param.mutable_dist_loss_param();
//   dist_loss_param->set_alpha(0.1);
//   dist_loss_param->set_beta(0.2);
//   dist_loss_param->set_kernel_size(3);
//   dist_loss_param->set_stride(1);
//   dist_loss_param->set_num_output(21);
//   DistLossLayer<Dtype> layer(layer_param);
//   layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
//   // the loss is just one number, summed and scaled for the entire batch
//   EXPECT_EQ(this->blob_top_loss_->num(), 1);
//   EXPECT_EQ(this->blob_top_loss_->channels(), 1);
//   EXPECT_EQ(this->blob_top_loss_->height(), 1);
//   EXPECT_EQ(this->blob_top_loss_->width(), 1);
//   //LOG(ERROR) << "this first test usually goes well";
// }

// TYPED_TEST(DistLossLayerTest, TestForward) {
//   this->TestForwardSquare();
//   //this->TestForwardRectHigh();
//   //this->TestForwardRectWide();
// }

// TYPED_TEST(DistLossLayerTest, TestBackward) {
//   this->TestBackwardSquare();
//   //this->TestForwardRectHigh();
//   //this->TestForwardRectWide();
// }

TYPED_TEST(DistLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  DistLossParameter* dist_loss_param = layer_param.mutable_dist_loss_param();
  dist_loss_param->set_alpha(0.5);
  dist_loss_param->set_beta(2);
  ConvolutionParameter* conv_param = layer_param.mutable_convolution_param();
  conv_param->set_kernel_h(2);
  conv_param->set_kernel_w(3);
  conv_param->set_stride_h(1);
  conv_param->set_stride_w(2);
  conv_param->set_pad_h(1);
  conv_param->set_pad_w(2);
  DistLossLayer<Dtype> layer(layer_param);
  // layer.SetUp(blob_bottom_vec_, blob_top_vec_);

  // //dist_loss_param->set_kernel_size(3);
  // dist_loss_param->set_kernel_h(2);
  // dist_loss_param->set_kernel_w(3);
  // //dist_loss_param->set_stride(2);
  // dist_loss_param->set_stride_h(2);
  // dist_loss_param->set_stride_w(1);
  // dist_loss_param->set_num_output(3);
  // DistLossLayer<Dtype> layer(layer_param);
  // We'll use the same criteria as in test_hinge_loss_layer
  GradientChecker<Dtype> checker(1e-3, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

// // TYPED_TEST(DistLossLayerTest, TestGradientWeights) {
// //   typedef typename TypeParam::Dtype Dtype;
// //   LayerParameter layer_param;
// //   DistLossParameter* dist_loss_param = layer_param.mutable_dist_loss_param();
// //   dist_loss_param->set_alpha(0.1);
// //   dist_loss_param->set_beta(10);
// //   dist_loss_param->set_kernel_size(3);
// //   dist_loss_param->set_stride(2);
// //   dist_loss_param->set_num_output(21);
// //   dist_loss_param->set_weight_source("../deeplab_scripts/voc12/loss_weight/loss_weight_trainval_aug.txt");
// //   DistLossLayer<Dtype> layer(layer_param);
// //   GradientChecker<Dtype> checker(1e-2, 1e-1, 1701);
// //   checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
// //       this->blob_top_vec_, 0);
// // }

// // TYPED_TEST(DistLossLayerTest, TestGradientWeights) {
// //   typedef typename TypeParam::Dtype Dtype;
// //   LayerParameter layer_param;
// //   // layer_param.mutable_dist_loss_param()->
// //   //   set_weight_source("examples/segnet/loss_weight_trainval_aug.txt");

// //    LayerParameter* dist_loss_param =
// //     dist_loss_param.mutable_dist_loss_param();
// //   dist_loss_param->set_kernel_size(3);
// //   dist_loss_param->set_stride(1);
// //   dist_loss_param->set_num_output(4);
// //   dist_loss_param->set_weight_source("../deeplab_scripts/voc12/loss_weight/loss_weight_trainval_aug.txt");
// //   dist_loss_param->set_ignore_label(255);
// //   shared_ptr<Layer<Dtype> > layer(
// //       new DistLossLayer<Dtype>(layer_param));
// //   // layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
// //   // layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);

// //   // layer_param->set_kernel_size(3);
// //   // layer_param->set_stride(1);
// //   // layer_param->set_num_output(4);
// //   // layer_param->set_weight_source("../deeplab_scripts/voc12/loss_weight/loss_weight_trainval_aug.txt");
// //   // layer_param->set_ignore_label(255);
// //   DistLossLayer<Dtype> layer(layer_param);
// //   GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
// //   checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
// //       this->blob_top_vec_, 0);
// // }

}  // namespace caffe

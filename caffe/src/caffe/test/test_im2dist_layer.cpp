#include <vector>
#include <cfloat>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/im2dist_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class Im2distLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
    // : blob_bottom_(new Blob<Dtype>(1, 1, 3, 4)),
  Im2distLayerTest()
    : blob_bottom_(new Blob<Dtype>(2, 2, 3, 4)),
      blob_top_(new Blob<Dtype>()) {
    // fill the values
    Caffe::set_random_seed(1701);
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);

    // for (int i = 0; i < 12; i += 4) {
    //   blob_bottom_->mutable_cpu_data()[i +  0] = 1;
    //   blob_bottom_->mutable_cpu_data()[i +  1] = 2;
    //   blob_bottom_->mutable_cpu_data()[i +  2] = 5;
    //   blob_bottom_->mutable_cpu_data()[i +  3] = 2;
    // }
  }
  virtual ~Im2distLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;

  void TestForwardSquare() {
    const int height = 3;
    const int width = 5;
    const int kernel = 3;
    const int pad = 1;
    const int stride = 1;
    const int num = 1;
    const int channels = 2;
    int height_col = (height + 2 * pad - kernel) / stride + 1;
    int width_col = (width + 2 * pad - kernel) / stride + 1;
    int channels_col = kernel * kernel;
    LayerParameter layer_param;
    ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
    convolution_param->add_kernel_size(kernel);
    convolution_param->add_stride(stride);
    convolution_param->add_pad(pad);
    Im2distParameter* im2dist_param =
      layer_param.mutable_im2dist_param();
    im2dist_param->set_norm(Im2distParameter_Norm_L1);
    bool remove_center = false;
    im2dist_param->set_remove_center(remove_center);
    bool remove_bounds = true;
    im2dist_param->set_remove_bounds(remove_bounds);
    
    blob_bottom_->Reshape(num, channels, height, width);
    // each channel looks like this: 
    //     [1 2 5 2 3]
    //     [9 4 1 4 8]
    //     [1 2 5 2 3]
    for (int i = 0; i < 15 * num * channels; i += 15) {
      blob_bottom_->mutable_cpu_data()[i +  0] = 1;
      blob_bottom_->mutable_cpu_data()[i +  1] = 2;
      blob_bottom_->mutable_cpu_data()[i +  2] = 5;
      blob_bottom_->mutable_cpu_data()[i +  3] = 2;
      blob_bottom_->mutable_cpu_data()[i +  4] = 3;
      blob_bottom_->mutable_cpu_data()[i +  5] = 9;
      blob_bottom_->mutable_cpu_data()[i +  6] = 4;
      blob_bottom_->mutable_cpu_data()[i +  7] = 1;
      blob_bottom_->mutable_cpu_data()[i +  8] = 4;
      blob_bottom_->mutable_cpu_data()[i +  9] = 8;
      blob_bottom_->mutable_cpu_data()[i + 10] = 1;
      blob_bottom_->mutable_cpu_data()[i + 11] = 2;
      blob_bottom_->mutable_cpu_data()[i + 12] = 5;
      blob_bottom_->mutable_cpu_data()[i + 13] = 2;
      blob_bottom_->mutable_cpu_data()[i + 14] = 3;
    }

    Im2distLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    EXPECT_EQ(blob_top_->num(), num);
    EXPECT_EQ(blob_top_->channels(), channels_col);
    EXPECT_EQ(blob_top_->height(), height_col);
    EXPECT_EQ(blob_top_->width(), width_col);
    layer.Forward(blob_bottom_vec_, blob_top_vec_);

    for (int i=0; i < height_col*width_col*channels_col; i++) {
      LOG(ERROR) << "top[" << i << "] = " << blob_top_->cpu_data()[i];
    }

  //   EXPECT_EQ(blob_top_->cpu_data()[0], 6);
  //   EXPECT_EQ(blob_top_->cpu_data()[1], 2);
  //   EXPECT_EQ(blob_top_->cpu_data()[2], 2);
  //   EXPECT_EQ(blob_top_->cpu_data()[3], 4);
  //   EXPECT_EQ(blob_top_->cpu_data()[4], 8);
  //   EXPECT_EQ(blob_top_->cpu_data()[5], 4);
  //   EXPECT_EQ(blob_top_->cpu_data()[6], 2);
  //   EXPECT_EQ(blob_top_->cpu_data()[7], 2);
  //   EXPECT_EQ(blob_top_->cpu_data()[8], 2);
  //   EXPECT_EQ(blob_top_->cpu_data()[9], 10);
  //   EXPECT_EQ(blob_top_->cpu_data()[10], 6);
  //   EXPECT_EQ(blob_top_->cpu_data()[11], 6);
  //   if (remove_center) {
  //     EXPECT_EQ(blob_top_->cpu_data()[12], FLT_MAX/10);
  //     EXPECT_EQ(blob_top_->cpu_data()[13], FLT_MAX/10);
  //     EXPECT_EQ(blob_top_->cpu_data()[14], FLT_MAX/10);
  //   } else {
  //     EXPECT_EQ(blob_top_->cpu_data()[12], 0);
  //     EXPECT_EQ(blob_top_->cpu_data()[13], 0);
  //     EXPECT_EQ(blob_top_->cpu_data()[14], 0);
  //   }
  //   EXPECT_EQ(blob_top_->cpu_data()[15], 6);
  //   EXPECT_EQ(blob_top_->cpu_data()[16], 6);
  //   EXPECT_EQ(blob_top_->cpu_data()[17], 8);
  //   EXPECT_EQ(blob_top_->cpu_data()[18], 6);
  //   EXPECT_EQ(blob_top_->cpu_data()[19], 2);
  //   EXPECT_EQ(blob_top_->cpu_data()[20], 2);
  //   EXPECT_EQ(blob_top_->cpu_data()[21], 4);
  //   EXPECT_EQ(blob_top_->cpu_data()[22], 8);
  //   EXPECT_EQ(blob_top_->cpu_data()[23], 4);
  //   EXPECT_EQ(blob_top_->cpu_data()[24], 2);
  //   EXPECT_EQ(blob_top_->cpu_data()[25], 2);
  //   EXPECT_EQ(blob_top_->cpu_data()[26], 2);
  }
};

TYPED_TEST_CASE(Im2distLayerTest, TestDtypesAndDevices);

TYPED_TEST(Im2distLayerTest, TestForward) {
  this->TestForwardSquare();
}

// TYPED_TEST(Im2distLayerTest, TestSetup) {
//   typedef typename TypeParam::Dtype Dtype;
//   LayerParameter layer_param;
//   ConvolutionParameter* convolution_param =
//       layer_param.mutable_convolution_param();
//   //convolution_param->add_kernel_size(3);
//   convolution_param->set_kernel_h(1);
//   convolution_param->set_kernel_w(2);
//   convolution_param->add_stride(2);
//   Im2distLayer<Dtype> layer(layer_param);
//   layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
//   EXPECT_EQ(this->blob_top_->num(), 2);
//   EXPECT_EQ(this->blob_top_->channels(), 2);
//   EXPECT_EQ(this->blob_top_->height(), 2);
//   EXPECT_EQ(this->blob_top_->width(), 2);
// }

TYPED_TEST(Im2distLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->set_kernel_h(3);
  convolution_param->set_kernel_w(3);
  convolution_param->set_stride_h(2);
  convolution_param->set_stride_w(2);
  convolution_param->set_pad_h(1);
  convolution_param->set_pad_w(0);
  // convolution_param->set_kernel_h(3);
  // convolution_param->set_kernel_w(3);
  // convolution_param->set_stride_h(1);
  // convolution_param->set_stride_w(1);
  // convolution_param->set_pad_h(1);
  // convolution_param->set_pad_w(1);
  Im2distParameter* im2dist_param =
    layer_param.mutable_im2dist_param();
  im2dist_param->set_norm(Im2distParameter_Norm_L2);
  im2dist_param->set_remove_center(false);
  Im2distLayer<Dtype> layer(layer_param);


  // //     [1 2 5 2]
  // //     [3 9 4 1]
  // //     [4 8 1 2]
  // for (int i = 0; i < 12 * 1 * 1; i += 12) {
  //   this->blob_bottom_->mutable_cpu_data()[i +  0] = 1;
  //   this->blob_bottom_->mutable_cpu_data()[i +  1] = 2;
  //   this->blob_bottom_->mutable_cpu_data()[i +  2] = 5;
  //   this->blob_bottom_->mutable_cpu_data()[i +  3] = 2;
  //   this->blob_bottom_->mutable_cpu_data()[i +  4] = 3;
  //   this->blob_bottom_->mutable_cpu_data()[i +  5] = 9;
  //   this->blob_bottom_->mutable_cpu_data()[i +  6] = 4;
  //   this->blob_bottom_->mutable_cpu_data()[i +  7] = 1;
  //   this->blob_bottom_->mutable_cpu_data()[i +  8] = 4;
  //   this->blob_bottom_->mutable_cpu_data()[i +  9] = 8;
  //   this->blob_bottom_->mutable_cpu_data()[i + 10] = 1;
  //   this->blob_bottom_->mutable_cpu_data()[i + 11] = 2;
  // }



  // // const int height = 3;
  // // const int width = 5;
  // // const int kernel = 3;
  // // const int pad = 0;
  // // const int stride = 1;
  // const int num = 1;
  // const int channels = 1;
  // int height_col = (height + 2 * pad - kernel) / stride + 1;
  // int width_col = (width + 2 * pad - kernel) / stride + 1;
  // int channels_col = kernel * kernel;

  //   for (int i = 0; i < 15 * num * channels; i += 15) {
  //     this->blob_bottom_->mutable_cpu_data()[i +  0] = 1;
  //     this->blob_bottom_->mutable_cpu_data()[i +  1] = 2;
  //     this->blob_bottom_->mutable_cpu_data()[i +  2] = 5;
  //     this->blob_bottom_->mutable_cpu_data()[i +  3] = 2;
  //     this->blob_bottom_->mutable_cpu_data()[i +  4] = 3;
  //     this->blob_bottom_->mutable_cpu_data()[i +  5] = 9;
  //     this->blob_bottom_->mutable_cpu_data()[i +  6] = 4;
  //     this->blob_bottom_->mutable_cpu_data()[i +  7] = 1;
  //     this->blob_bottom_->mutable_cpu_data()[i +  8] = 4;
  //     this->blob_bottom_->mutable_cpu_data()[i +  9] = 8;
  //     this->blob_bottom_->mutable_cpu_data()[i + 10] = 1;
  //     this->blob_bottom_->mutable_cpu_data()[i + 11] = 2;
  //     this->blob_bottom_->mutable_cpu_data()[i + 12] = 5;
  //     this->blob_bottom_->mutable_cpu_data()[i + 13] = 2;
  //     this->blob_bottom_->mutable_cpu_data()[i + 14] = 3;
  //   }

  GradientChecker<Dtype> checker(1e-3, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);

  // // for (int i=0; i < height_col*width_col*channels_col; i++) {
  // for (int i=0; i < this->blob_top_->count(); i++) {
  //   LOG(ERROR) << "top[" << i << "] = " << this->blob_top_->cpu_data()[i];
  // }

  // for (int i=0; i < this->blob_top_->count(); i++) {
  //   LOG(ERROR) << "top_diff[" << i << "] = " << this->blob_top_->cpu_diff()[i];
  // }

  // // // for (int i=0; i < height*width*channels; i++) {
  // for (int i=0; i < this->blob_bottom_->count(); i++) {
  //   LOG(ERROR) << "bottom_diff[" << i << "] = " << this->blob_bottom_->cpu_diff()[i];
  // }


}

// TYPED_TEST(Im2distLayerTest, TestDilatedGradient) {
//   typedef typename TypeParam::Dtype Dtype;
//   LayerParameter layer_param;
//   ConvolutionParameter* convolution_param =
//       layer_param.mutable_convolution_param();

//   vector<int> bottom_shape;
//   bottom_shape.push_back(3);
//   bottom_shape.push_back(2);
//   bottom_shape.push_back(5);
//   bottom_shape.push_back(6);
//   this->blob_bottom_->Reshape(bottom_shape);
//   // since we reshaped, we need to refill
//   Caffe::set_random_seed(1701);
//   FillerParameter filler_param;
//   GaussianFiller<Dtype> filler(filler_param);
//   filler.Fill(this->blob_bottom_);

//   convolution_param->add_pad(1);
//   convolution_param->set_kernel_h(2);
//   convolution_param->set_kernel_w(3);
//   convolution_param->set_stride_h(2);
//   convolution_param->set_stride_w(3);
//   convolution_param->add_dilation(2);
//   Im2distParameter* im2dist_param =
//     layer_param.mutable_im2dist_param();
//   im2dist_param->set_norm(Im2distParameter_Norm_L2);
//   Im2distLayer<Dtype> layer(layer_param);
//   GradientChecker<Dtype> checker(1e-3, 1e-2);
//   checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
//       this->blob_top_vec_);
// }

}  // namespace caffe

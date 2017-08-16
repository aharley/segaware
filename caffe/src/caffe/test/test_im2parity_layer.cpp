#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/im2parity_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class Im2parityLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  Im2parityLayerTest()
    : blob_bottom_(new Blob<Dtype>(1, 1, 3, 5)),
      blob_top_(new Blob<Dtype>()) {
    // fill the values
    Caffe::set_random_seed(1701);
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);

    for (int i = 0; i < 12; i += 4) {
      blob_bottom_->mutable_cpu_data()[i +  0] = 1;
      blob_bottom_->mutable_cpu_data()[i +  1] = 2;
      blob_bottom_->mutable_cpu_data()[i +  2] = 5;
      blob_bottom_->mutable_cpu_data()[i +  3] = 2;
    }
  }
  virtual ~Im2parityLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;

  void TestForwardSquare() {
    const int height = 3;
    const int width = 5;
    const int kernel = 3;
    const int pad = 0;
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
    
    blob_bottom_->Reshape(num, channels, height, width);
    // each channel looks like this: 
    //     [1 2 5 2 3]
    //     [9 1 1 1 8]
    //     [1 2 5 2 3]
    for (int i = 0; i < 15 * num * channels; i += 15) {
      blob_bottom_->mutable_cpu_data()[i +  0] = 1;
      blob_bottom_->mutable_cpu_data()[i +  1] = 2;
      blob_bottom_->mutable_cpu_data()[i +  2] = 5;
      blob_bottom_->mutable_cpu_data()[i +  3] = 2;
      blob_bottom_->mutable_cpu_data()[i +  4] = 3;
      blob_bottom_->mutable_cpu_data()[i +  5] = 9;
      blob_bottom_->mutable_cpu_data()[i +  6] = 1;
      blob_bottom_->mutable_cpu_data()[i +  7] = 1;
      blob_bottom_->mutable_cpu_data()[i +  8] = 1;
      blob_bottom_->mutable_cpu_data()[i +  9] = 8;
      blob_bottom_->mutable_cpu_data()[i + 10] = 1;
      blob_bottom_->mutable_cpu_data()[i + 11] = 2;
      blob_bottom_->mutable_cpu_data()[i + 12] = 5;
      blob_bottom_->mutable_cpu_data()[i + 13] = 2;
      blob_bottom_->mutable_cpu_data()[i + 14] = 3;
    }

    Im2parityLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    EXPECT_EQ(blob_top_->num(), num);
    EXPECT_EQ(blob_top_->channels(), channels_col);
    EXPECT_EQ(blob_top_->height(), height_col);
    EXPECT_EQ(blob_top_->width(), width_col);
    layer.Forward(blob_bottom_vec_, blob_top_vec_);

    // for (int i=0; i < height_col*width_col*channels_col; i++) {
    //   LOG(ERROR) << "top[" << i << "] = " << blob_top_->cpu_data()[i];
    // }
    EXPECT_EQ(blob_top_->cpu_data()[0], 1);
    EXPECT_EQ(blob_top_->cpu_data()[1], 0);
    EXPECT_EQ(blob_top_->cpu_data()[2], 0);
    EXPECT_EQ(blob_top_->cpu_data()[3], 0);
    EXPECT_EQ(blob_top_->cpu_data()[4], 0);
    EXPECT_EQ(blob_top_->cpu_data()[5], 0);
    EXPECT_EQ(blob_top_->cpu_data()[6], 0);
    EXPECT_EQ(blob_top_->cpu_data()[7], 0);
    EXPECT_EQ(blob_top_->cpu_data()[8], 0);
    EXPECT_EQ(blob_top_->cpu_data()[9], 0);
    EXPECT_EQ(blob_top_->cpu_data()[10], 1);
    EXPECT_EQ(blob_top_->cpu_data()[11], 1);
    EXPECT_EQ(blob_top_->cpu_data()[12], 1);
    EXPECT_EQ(blob_top_->cpu_data()[13], 1);
    EXPECT_EQ(blob_top_->cpu_data()[14], 1);
    EXPECT_EQ(blob_top_->cpu_data()[15], 1);
    EXPECT_EQ(blob_top_->cpu_data()[16], 1);
    EXPECT_EQ(blob_top_->cpu_data()[17], 0);
    EXPECT_EQ(blob_top_->cpu_data()[18], 1);
    EXPECT_EQ(blob_top_->cpu_data()[19], 0);
    EXPECT_EQ(blob_top_->cpu_data()[20], 0);
    EXPECT_EQ(blob_top_->cpu_data()[21], 0);
    EXPECT_EQ(blob_top_->cpu_data()[22], 0);
    EXPECT_EQ(blob_top_->cpu_data()[23], 0);
    EXPECT_EQ(blob_top_->cpu_data()[24], 0);
    EXPECT_EQ(blob_top_->cpu_data()[25], 0);
    EXPECT_EQ(blob_top_->cpu_data()[26], 0);
  }
};

TYPED_TEST_CASE(Im2parityLayerTest, TestDtypesAndDevices);

TYPED_TEST(Im2parityLayerTest, TestForward) {
  this->TestForwardSquare();
}

// TYPED_TEST(Im2parityLayerTest, TestSetup) {
//   typedef typename TypeParam::Dtype Dtype;
//   LayerParameter layer_param;
//   ConvolutionParameter* convolution_param =
//       layer_param.mutable_convolution_param();
//   //convolution_param->add_kernel_size(3);
//   convolution_param->set_kernel_h(1);
//   convolution_param->set_kernel_w(2);
//   convolution_param->add_stride(2);
//   Im2parityLayer<Dtype> layer(layer_param);
//   layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
//   EXPECT_EQ(this->blob_top_->num(), 2);
//   EXPECT_EQ(this->blob_top_->channels(), 27);
//   EXPECT_EQ(this->blob_top_->height(), 2);
//   EXPECT_EQ(this->blob_top_->width(), 2);
// }


}  // namespace caffe

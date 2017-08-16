#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/channel_reduce_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class ChannelReduceLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  ChannelReduceLayerTest()
    : blob_bottom_(new Blob<Dtype>(2, 6, 5, 3)),
      blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);

    // for (int i = 0; i < 6*1*2; i += 4) {
    //   blob_bottom_->mutable_cpu_data()[i +  0] = 1;
    //   blob_bottom_->mutable_cpu_data()[i +  1] = 2;
    //   blob_bottom_->mutable_cpu_data()[i +  2] = 5;
    //   blob_bottom_->mutable_cpu_data()[i +  3] = 2;
    //   // blob_bottom_->mutable_cpu_data()[i +  4] = 3;
    // }
  }
  virtual ~ChannelReduceLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;

  void TestForwardSquare() {
    const int height = 3;
    const int width = 5;
    const int num = 1;
    const int channels = 4;
    const int num_channels = 2;
    LayerParameter layer_param;
    ChannelReduceParameter* channel_reduce_param =
      layer_param.mutable_channel_reduce_param();
    channel_reduce_param->set_num_channels(num_channels);
    channel_reduce_param->set_operation(ChannelReduceParameter_Op_MAX);
    
    blob_bottom_->Reshape(num, channels, height, width);
    // image has four channels of this:
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
    // except we'll set the first channel to all threes
    for (int i=0; i < 1*height*width; ++i) {
      blob_bottom_->mutable_cpu_data()[i] = 3;
    }
    // and the last channel to all nines
    for (int i=(channels-1)*height*width; i < channels*height*width; ++i) {
      blob_bottom_->mutable_cpu_data()[i] = 9;
    }
    
    ChannelReduceLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    EXPECT_EQ(blob_top_->num(), num);
    EXPECT_EQ(blob_top_->channels(), num_channels);
    EXPECT_EQ(blob_top_->height(), height);
    EXPECT_EQ(blob_top_->width(), width);

    layer.Forward(blob_bottom_vec_, blob_top_vec_);
    // for (int i=0; i < height*width*channels; i++) {
    //   LOG(ERROR) << "bottom[" << i << "] = " << blob_bottom_->cpu_data()[i];
    // }
    // for (int i=0; i < height*width*num_channels; i++) {
    //   LOG(ERROR) << "top[" << i << "] = " << blob_top_->cpu_data()[i];
    // }
    for (int chan = 0; chan < num_channels; chan++) {
      for (int i = chan*height*width; i < height*width*(chan+1); i++) {
    	if (chan==0) 
    	  EXPECT_EQ(blob_top_->cpu_data()[i],std::max(blob_bottom_->cpu_data()[i+height*width],Dtype(3.0)));
	else if (chan==1)
    	  EXPECT_EQ(blob_top_->cpu_data()[i],9);
      }
    }
    // for (int i = 0; i < height*width*num_channels; i += height*width) {
    //   EXPECT_EQ(blob_top_->cpu_data()[i +  0] = 1;
    //   EXPECT_EQ(blob_top_->cpu_data()[i +  1] = 2;
    //   EXPECT_EQ(blob_top_->cpu_data()[i +  2] = 5;
    //   EXPECT_EQ(blob_top_->cpu_data()[i +  3] = 2;
    //   EXPECT_EQ(blob_top_->cpu_data()[i +  4] = 3;

    //   EXPECT_EQ(blob_top_->cpu_data()[i +  5] = 9;
    //   EXPECT_EQ(blob_top_->cpu_data()[i +  6] = 4;
    //   EXPECT_EQ(blob_top_->cpu_data()[i +  7] = 1;
    //   EXPECT_EQ(blob_top_->cpu_data()[i +  8] = 4;
    //   EXPECT_EQ(blob_top_->cpu_data()[i +  9] = 8;

    //   EXPECT_EQ(blob_top_->cpu_data()[i + 10] = 1;
    //   EXPECT_EQ(blob_top_->cpu_data()[i + 11] = 2;
    //   EXPECT_EQ(blob_top_->cpu_data()[i + 12] = 5;
    //   EXPECT_EQ(blob_top_->cpu_data()[i + 13] = 2;
    //   EXPECT_EQ(blob_top_->cpu_data()[i + 14] = 3;
    // }
  }
};

TYPED_TEST_CASE(ChannelReduceLayerTest, TestDtypesAndDevices);

TYPED_TEST(ChannelReduceLayerTest, TestForward) {
  this->TestForwardSquare();
}

TYPED_TEST(ChannelReduceLayerTest, TestSumGradient) {
  typedef typename TypeParam::Dtype Dtype;
  const int num_channels = 3;
  LayerParameter layer_param;
  ChannelReduceParameter* channel_reduce_param =
    layer_param.mutable_channel_reduce_param();
  channel_reduce_param->set_num_channels(num_channels);
  channel_reduce_param->set_operation(ChannelReduceParameter_Op_SUM);
  ChannelReduceLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-4, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(ChannelReduceLayerTest, TestMaxGradient) {
  typedef typename TypeParam::Dtype Dtype;
  const int num_channels = 3;
  LayerParameter layer_param;
  ChannelReduceParameter* channel_reduce_param =
    layer_param.mutable_channel_reduce_param();
  channel_reduce_param->set_num_channels(num_channels);
  channel_reduce_param->set_operation(ChannelReduceParameter_Op_MAX);
  ChannelReduceLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-4, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

}  // namespace caffe

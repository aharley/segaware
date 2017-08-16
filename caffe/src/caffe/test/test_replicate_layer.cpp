#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/replicate_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class ReplicateLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

protected:
  ReplicateLayerTest()
    : blob_bottom_(new Blob<Dtype>(2, 3, 6, 5)),
      blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    shared_ptr<ConstantFiller<Dtype> > filler;
    FillerParameter filler_param;
    filler_param.set_value(1.);
    filler.reset(new ConstantFiller<Dtype>(filler_param));
    filler->Fill(this->blob_bottom_);
    filler_param.set_value(2.);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~ReplicateLayerTest() {
    delete blob_bottom_;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(ReplicateLayerTest, TestDtypesAndDevices);

TYPED_TEST(ReplicateLayerTest, TestSetupNum) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_replicate_param()->set_axis(0);
  layer_param.mutable_replicate_param()->set_num_copies(2);
  ReplicateLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(),
	    this->blob_bottom_->num() + this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_->height());
  EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_->width());
}

TYPED_TEST(ReplicateLayerTest, TestSetupChannels) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_replicate_param()->set_num_copies(2);
  ReplicateLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(),
	    this->blob_bottom_->channels() + this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_->height());
  EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_->width());
}

TYPED_TEST(ReplicateLayerTest, TestSetupChannelsNegativeIndexing) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_replicate_param()->set_num_copies(2);
  ReplicateLayer<Dtype> layer(layer_param);
  // "channels" index is the third one from the end -- test negative indexing
  // by setting axis to -3 and checking that we get the same results as above in
  // TestSetupChannels.
  layer_param.mutable_replicate_param()->set_axis(-3);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(),
	    this->blob_bottom_->channels() + this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_->height());
  EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_->width());
}

TYPED_TEST(ReplicateLayerTest, TestForwardNum) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_replicate_param()->set_num_copies(2);
  layer_param.mutable_replicate_param()->set_axis(0);
  ReplicateLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int n = 0; n < this->blob_bottom_vec_[0]->num(); ++n) {
    for (int c = 0; c < this->blob_top_->channels(); ++c) {
      for (int h = 0; h < this->blob_top_->height(); ++h) {
	for (int w = 0; w < this->blob_top_->width(); ++w) {
	  EXPECT_EQ(this->blob_top_->data_at(n, c, h, w),
		    this->blob_bottom_vec_[0]->data_at(n, c, h, w));
	}
      }
    }
  }
  for (int n = 0; n < this->blob_bottom_vec_[0]->num(); ++n) {
    for (int c = 0; c < this->blob_top_->channels(); ++c) {
      for (int h = 0; h < this->blob_top_->height(); ++h) {
	for (int w = 0; w < this->blob_top_->width(); ++w) {
	  EXPECT_EQ(this->blob_top_->data_at(n + 2, c, h, w),
		    this->blob_bottom_vec_[0]->data_at(n, c, h, w));
	}
      }
    }
  }
}

TYPED_TEST(ReplicateLayerTest, TestForwardChannels) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_replicate_param()->set_num_copies(2);
  ReplicateLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int n = 0; n < this->blob_top_->num(); ++n) {
    for (int c = 0; c < this->blob_bottom_->channels(); ++c) {
      for (int h = 0; h < this->blob_top_->height(); ++h) {
	for (int w = 0; w < this->blob_top_->width(); ++w) {
	  EXPECT_EQ(this->blob_top_->data_at(n, c, h, w),
		    this->blob_bottom_vec_[0]->data_at(n, c, h, w));
	}
      }
    }
    for (int c = 0; c < this->blob_bottom_->channels(); ++c) {
      for (int h = 0; h < this->blob_top_->height(); ++h) {
	for (int w = 0; w < this->blob_top_->width(); ++w) {
	  EXPECT_EQ(this->blob_top_->data_at(n, c + 3, h, w),
		    this->blob_bottom_vec_[0]->data_at(n, c, h, w));
	}
      }
    }
  }
}

TYPED_TEST(ReplicateLayerTest, TestGradientNum) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_replicate_param()->set_num_copies(2);
  layer_param.mutable_replicate_param()->set_axis(0);
  ReplicateLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradient(&layer, this->blob_bottom_vec_,
			this->blob_top_vec_);
}

TYPED_TEST(ReplicateLayerTest, TestGradientChannels) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_replicate_param()->set_num_copies(2);
  ReplicateLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradient(&layer, this->blob_bottom_vec_,
			this->blob_top_vec_);
}

}  // namespace caffe

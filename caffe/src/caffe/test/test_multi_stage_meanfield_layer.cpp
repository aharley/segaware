#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/multi_stage_meanfield_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class MultiStageMeanfieldLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

protected:
  MultiStageMeanfieldLayerTest()
    : blob_bottom_(new Blob<Dtype>(1, 2, 3, 4)),
      blob_bottom_2_(new Blob<Dtype>(1, 2, 3, 4)),
      blob_bottom_3_(new Blob<Dtype>(1, 3, 3, 4)),
      blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_value(1.);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    filler.Fill(this->blob_bottom_2_);
    filler.Fill(this->blob_bottom_3_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_2_);
    blob_bottom_vec_.push_back(blob_bottom_3_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~MultiStageMeanfieldLayerTest() {
    delete blob_bottom_;
    delete blob_bottom_2_;
    delete blob_bottom_3_;
    delete blob_top_;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_2_;
  Blob<Dtype>* const blob_bottom_3_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MultiStageMeanfieldLayerTest, TestDtypesAndDevices);

TYPED_TEST(MultiStageMeanfieldLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MultiStageMeanfieldParameter* multi_stage_meanfield_param =
      layer_param.mutable_multi_stage_meanfield_param();
  multi_stage_meanfield_param->set_num_iterations(2);
  multi_stage_meanfield_param->set_theta_alpha(3);
  multi_stage_meanfield_param->set_theta_beta(3);
  multi_stage_meanfield_param->set_theta_gamma(3);

  LOG(ERROR) << "set up params";

  // int num = 2;
  // int img_channels = 2;
  // int emb_channels = 1;
  // int height = 3;
  // int width = 4;
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

  // // for (int i = 0; i < 9 * num * img_channels; i += 9) {
  // //   this->blob_bottom_->mutable_cpu_data()[i +  0] = 10;
  // //   this->blob_bottom_->mutable_cpu_data()[i +  1] = 2;
  // //   this->blob_bottom_->mutable_cpu_data()[i +  2] = 5;
  // //   this->blob_bottom_->mutable_cpu_data()[i +  3] = 2;
  // //   this->blob_bottom_->mutable_cpu_data()[i +  4] = 3;
  // //   this->blob_bottom_->mutable_cpu_data()[i +  5] = 9;
  // //   this->blob_bottom_->mutable_cpu_data()[i +  6] = 1;
  // //   this->blob_bottom_->mutable_cpu_data()[i +  7] = 1 + i;
  // //   this->blob_bottom_->mutable_cpu_data()[i +  8] = 1 + i;
  // //   // this->blob_bottom_->mutable_cpu_data()[i +  9] = 8;
  // //   // this->blob_bottom_->mutable_cpu_data()[i + 10] = 1;
  // //   // this->blob_bottom_->mutable_cpu_data()[i + 11] = 2;
  // //   // this->blob_bottom_->mutable_cpu_data()[i + 12] = 5;
  // //   // this->blob_bottom_->mutable_cpu_data()[i + 13] = 2;
  // //   // this->blob_bottom_->mutable_cpu_data()[i + 14] = 3;
  // // }
  // // for (int i = 0; i < 9 * num * emb_channels; i += 9) {
  // //   this->blob_bottom_2_->mutable_cpu_data()[i +  0] = 15;
  // //   this->blob_bottom_2_->mutable_cpu_data()[i +  1] = 20;
  // //   this->blob_bottom_2_->mutable_cpu_data()[i +  2] = 50;
  // //   this->blob_bottom_2_->mutable_cpu_data()[i +  3] = 2;
  // //   this->blob_bottom_2_->mutable_cpu_data()[i +  4] = 3;
  // //   this->blob_bottom_2_->mutable_cpu_data()[i +  5] = 9;
  // //   this->blob_bottom_2_->mutable_cpu_data()[i +  6] = 1;
  // //   this->blob_bottom_2_->mutable_cpu_data()[i +  7] = 9;
  // //   this->blob_bottom_2_->mutable_cpu_data()[i +  8] = 10;
  // //   // this->blob_bottom_2_->mutable_cpu_data()[i +  9] = 8;
  // //   // this->blob_bottom_2_->mutable_cpu_data()[i + 10] = 1;
  // //   // this->blob_bottom_2_->mutable_cpu_data()[i + 11] = 2;
  // //   // this->blob_bottom_2_->mutable_cpu_data()[i + 12] = 5;
  // //   // this->blob_bottom_2_->mutable_cpu_data()[i + 13] = 2;
  // //   // this->blob_bottom_2_->mutable_cpu_data()[i + 14] = 3;
  // // }
  // for (int i = 0; i < height*width*num*img_channels; i++) {
  //   this->blob_bottom_->mutable_cpu_data()[i] = i*i;
  // }
  // for (int i = 0; i < height*width*num*emb_channels; i++) {
  //   this->blob_bottom_2_->mutable_cpu_data()[i] = i*i;
  // }

  // LOG(ERROR) << "ok got it";
  // const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  // for (int i=0; i < this->blob_bottom_->count(); i++) {
  //   LOG(ERROR) << "bottom[" << i << "] = " << bottom_data[i];
  // }
  // const Dtype* bottom2_data = this->blob_bottom_2_->cpu_data();
  // for (int i=0; i < this->blob_bottom_2_->count(); i++) {
  //   LOG(ERROR) << "bottom2[" << i << "] = " << bottom2_data[i];
  // }

  MultiStageMeanfieldLayer<Dtype> layer(layer_param);

  // layer.blobs().resize(3);
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

  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  
  const Dtype* top_data = this->blob_top_->cpu_data();
  for (int i=0; i < this->blob_top_->count(); i++) {
    LOG(ERROR) << "top[" << i << "] = " << top_data[i];
  }
  GradientChecker<Dtype> checker(1e-4, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
  // just check backprop to the first bottom blob (the unaries)
  // checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
  // 	  this->blob_top_vec_, 0);
}




}  // namespace caffe

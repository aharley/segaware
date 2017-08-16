#include <cfloat>
#include <vector>

#include "caffe/layers/channel_reduce_layer.hpp"

namespace caffe {

template <typename Dtype>
void ChannelReduceLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  ChannelReduceParameter channel_reduce_param = this->layer_param_.channel_reduce_param();
  if (channel_reduce_param.has_num_channels())
    num_channels_ = this->layer_param_.channel_reduce_param().num_channels();
  else 
    num_channels_ = 1;

  CHECK_GT(num_channels_, 0) << "num_channels must be greater than 0.";
  channels_ = bottom[0]->channels();
  CHECK((num_channels_ % channels_)) << "num_channels must be a divisor of channels.";

  op_ = this->layer_param_.channel_reduce_param().operation();
}

template <typename Dtype>
void ChannelReduceLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
  //     << "corresponding to (num, channels, height, width)";
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  channels_ = bottom[0]->channels();
  top[0]->Reshape(bottom[0]->num(), num_channels_, height_, width_);
  max_idx_.Reshape(bottom[0]->num(), num_channels_, height_, width_);
}

template <typename Dtype>
void ChannelReduceLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = top[0]->height()*top[0]->width(); // number of pixels in a channel
  int* mask = max_idx_.mutable_cpu_data();
  switch (op_) {
  case ChannelReduceParameter_Op_SUM:
    caffe_set(top[0]->count(), Dtype(0), top_data);
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
	int c_top = c / (channels_/num_channels_);
	caffe_add(count, 
		  bottom_data + bottom[0]->offset(n,c), 
		  top_data + top[0]->offset(n,c_top),
		  top_data + top[0]->offset(n,c_top));
      }
    }
    break;
  case ChannelReduceParameter_Op_MAX:
    caffe_set(top[0]->count(), Dtype(-FLT_MAX), top_data);
    caffe_set(top[0]->count(), -1, mask);
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
	int c_top = c / (channels_/num_channels_);
	for (int h = 0; h < height_; ++h) {
	  for (int w = 0; w < width_; ++w) {
	    int bot_index = (c * height_ + h) * width_ + w;
	    int top_index = (c_top * height_ + h) * width_ + w;
	    if (bottom_data[bot_index] > top_data[top_index]) {
	      top_data[top_index] = bottom_data[bot_index];
	      mask[top_index] = bot_index;
	    }
	  }
	}
      }
      bottom_data += bottom[0]->offset(1);
      top_data += top[0]->offset(1);
      mask += top[0]->offset(1);
    }
    break;
  default:
    LOG(FATAL) << "Unknown operation.";
  }
  // Dtype* bot_d = bottom[0]->mutable_cpu_data();
  // for (int i=0; i < 12; i++) {
  //   LOG(ERROR) << "okforward bottom_data[" << i << "] = " << bot_d[i];
  // }
}

template <typename Dtype>
void ChannelReduceLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
				      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  // Dtype* bot_d = bottom[0]->mutable_cpu_data();
  // for (int i=0; i < bottom[0]->count(); i++) {
  //   LOG(ERROR) << "forward bottom_data[" << i << "] = " << bot_d[i];
  // }
  // Dtype* top_d = top[0]->mutable_cpu_data();
  // for (int i=0; i < top[0]->count(); i++) {
  //   LOG(ERROR) << "forward top_data[" << i << "] = " << top_d[i];
  // }

  const int count = bottom[0]->height()*bottom[0]->width(); // number of pixels in a channel
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const int* mask = max_idx_.cpu_data();
  switch (op_) {
  case ChannelReduceParameter_Op_SUM:
    // i need to walk through the channels in bottom_diff,
    // and at each channel, copy in the appropriate top_diff
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
	int c_top = c / (channels_/num_channels_);
	caffe_copy(count, 
		   top_diff+top[0]->offset(n,c_top), 
		   bottom_diff+bottom[0]->offset(n,c));
      }
    }
    break;
  case ChannelReduceParameter_Op_MAX:
    caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
    	int c_top = c / (channels_/num_channels_);
    	for (int h = 0; h < height_; ++h) {
    	  for (int w = 0; w < width_; ++w) {
    	    int bot_index = (c * height_ + h) * width_ + w;
    	    int top_index = (c_top * height_ + h) * width_ + w;
    	    if (bot_index == mask[top_index]) {
    	      bottom_diff[bot_index] += top_diff[top_index];
	      // LOG(ERROR) << "adding " << top_diff[top_index] << " to index " << top_index;
    	    }
    	  }
    	}
      }
      bottom_diff += bottom[0]->offset(1);
      top_diff += top[0]->offset(1);
      mask += top[0]->offset(1);
    }
    break;
  default:
    LOG(FATAL) << "Unknown operation.";
  }
  // Dtype* t_diff = top[0]->mutable_cpu_diff();
  // for (int i=0; i < top[0]->count(); i++) {
  //   LOG(ERROR) << "backward top_diff[" << i << "] = " << t_diff[i];
  // }
  // Dtype* bot_diff = bottom[0]->mutable_cpu_diff();
  // for (int i=0; i < bottom[0]->count(); i++) {
  //   LOG(ERROR) << "backward bottom_diff[" << i << "] = " << bot_diff[i];
  // }
}

#ifdef CPU_ONLY
STUB_GPU(ChannelReduceLayer);
#endif

INSTANTIATE_CLASS(ChannelReduceLayer);
REGISTER_LAYER_CLASS(ChannelReduce);

}  // namespace caffe

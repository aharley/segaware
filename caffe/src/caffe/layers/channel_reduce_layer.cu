#include <cfloat>
#include <vector>

#include "caffe/layers/channel_reduce_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void MaxChannelReduceForward(const int n, const Dtype* bot_data,
    const int height, const int width, const int channels, const int num_channels, 
    Dtype* top_data, int* mask) {
  CUDA_KERNEL_LOOP(index, n) {
    //Dtype val = 0;
    int w = index % width;
    int h = (index / width) % height;
    int c = index / (width * height);
    // w,h,c represent where we are in top_data
    int c_to_channel_reduce = channels/num_channels;
    // say we're at c=0; i want to grab chans 0..c_to_channel_reduce
    int coeff_h_col = (1 - height) * width;
    int coeff_w_col = 1 - height * width;
    for (int chan = c*c_to_channel_reduce; chan < (c+1)*c_to_channel_reduce; ++chan) {
      int offset = (chan + h + w) * height * width;
      int bot_index = offset + h * coeff_h_col + w * coeff_w_col;
      if (bot_data[bot_index]>top_data[index]) {
	top_data[index] = bot_data[bot_index];
	mask[index] = bot_index;
      }
    }
  }
}

template <typename Dtype>
void ChannelReduceLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  caffe_gpu_set(top[0]->count(), Dtype(0), top_data);
  const int count = top[0]->height()*top[0]->width(); // number of pixels in a channel
  int* mask = max_idx_.mutable_gpu_data();

  switch (op_) {
  case ChannelReduceParameter_Op_SUM:
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
	int c_top = c / (channels_/num_channels_);
	caffe_gpu_add(count, 
		      bottom_data + bottom[0]->offset(n,c), 
		      top_data + top[0]->offset(n,c_top),
		      top_data + top[0]->offset(n,c_top));
      }
    }
    break;
  case ChannelReduceParameter_Op_MAX:
    caffe_gpu_set(top[0]->count(), Dtype(-FLT_MAX), top_data);
    for (int n = 0; n < bottom[0]->num(); ++n) {
      int num_kernels = num_channels_ * height_ * width_;
      MaxChannelReduceForward<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
	CAFFE_CUDA_NUM_THREADS>>>(num_kernels, 
				  bottom_data+bottom[0]->offset(n), 
				  height_, width_, channels_, num_channels_, 
				  top_data+top[0]->offset(n),
				  mask+top[0]->offset(n));
      CUDA_POST_KERNEL_CHECK;
    }
    break;
  default:
    LOG(FATAL) << "Unknown operation.";
  }
  // const Dtype* embs = top[0]->cpu_data();
  // for (int i=0; i < 10; i++){
  //   LOG(ERROR) << "for example, cr[" << i << "] = " << embs[i];
  // }
}

template <typename Dtype>
__global__ void MaxChannelReduceBackward(const int n, const Dtype* top_diff,
    const int height, const int width, const int channels, const int num_channels, 
    Dtype* bot_diff, const int* mask) {
  CUDA_KERNEL_LOOP(index, n) {
    //Dtype val = 0;
    int w = index % width;
    int h = (index / width) % height;
    int c = index / (width * height);
    // w,h,c represent where we are in bottom_data
    // this index may or may not have contributed to the top data
    int c_top = c / (channels/num_channels);
    int top_index = (c_top * height + h) * width + w;
    if (index == mask[top_index]) {
      bot_diff[index] += top_diff[top_index];
    }
  }
}


template <typename Dtype>
void ChannelReduceLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int count = bottom[0]->height()*bottom[0]->width(); // number of pixels in a channel
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int* mask = max_idx_.gpu_data();

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
  // i need to walk through the channels in bottom_diff,
  // and at each channel, copy in the appropriate top_diff
  caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom_diff);
  for (int n = 0; n < bottom[0]->num(); ++n) {
    int num_kernels = channels_ * height_ * width_;
    MaxChannelReduceBackward<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
      CAFFE_CUDA_NUM_THREADS>>>(num_kernels, 
				top_diff+top[0]->offset(n), 
				height_, width_, channels_, num_channels_, 
				bottom_diff+bottom[0]->offset(n),
				mask+top[0]->offset(n));
    CUDA_POST_KERNEL_CHECK;
  }
    break;
  default:
    LOG(FATAL) << "Unknown operation.";
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ChannelReduceLayer);

}  // namespace caffe

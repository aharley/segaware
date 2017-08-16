#include <vector>

#include "caffe/layers/im2dist_layer.hpp"
#include "caffe/util/im2dist.hpp"

namespace caffe {

template <typename Dtype>
void Im2distLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
				      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* diff_data = diff_.mutable_gpu_data();
  const int num_kernels = channels_ * top[0]->count(channel_axis_ + 1);
  for (int n = 0; n < num_; ++n) {
    im2dist_gpu(bottom_data + n * bottom_dim_, channels_,
		bottom[0]->shape(channel_axis_ + 1),
		bottom[0]->shape(channel_axis_ + 2),
		kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
		pad_.cpu_data()[0], pad_.cpu_data()[1],
		stride_.cpu_data()[0], stride_.cpu_data()[1],
		dilation_.cpu_data()[0], dilation_.cpu_data()[1],
		top_data + n * top_dim_,
		diff_data + n * diff_dim_, norm_,
		remove_center_, remove_bounds_);
  }
  // const Dtype* embs = top[0]->cpu_data();
  // // for (int i=190; i < 210; i++){
  // for (int i=0; i < top[0]->count(); i++) {
  //   LOG(ERROR) << "for example, im2dist[" << i << "] = " << embs[i];
  // }
}

template <typename Dtype>
void Im2distLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* diff_data = diff_.gpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  for (int n = 0; n < num_; ++n) {
    dist2im_gpu(top_diff + n * top_dim_, 
  		diff_data + n * diff_dim_, 
  		channels_,
  		bottom[0]->shape(channel_axis_ + 1),
  		bottom[0]->shape(channel_axis_ + 2),
  		kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
  		pad_.cpu_data()[0], pad_.cpu_data()[1],
  		stride_.cpu_data()[0], stride_.cpu_data()[1],
  		dilation_.cpu_data()[0], dilation_.cpu_data()[1],
  		bottom_diff + n * bottom_dim_, norm_,
		remove_center_, remove_bounds_);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(Im2distLayer);

}  // namespace caffe

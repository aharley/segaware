#include <vector>

#include "caffe/layers/im2parity_layer.hpp"
#include "caffe/util/im2parity.hpp"

namespace caffe {

template <typename Dtype>
void Im2parityLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
				      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int num_kernels = channels_ * top[0]->count(channel_axis_ + 1);
  for (int n = 0; n < num_; ++n) {
    im2parity_gpu(bottom_data + n * bottom_dim_, channels_,
  		bottom[0]->shape(channel_axis_ + 1),
  		bottom[0]->shape(channel_axis_ + 2),
  		kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
  		pad_.cpu_data()[0], pad_.cpu_data()[1],
  		stride_.cpu_data()[0], stride_.cpu_data()[1],
  		dilation_.cpu_data()[0], dilation_.cpu_data()[1],
  		has_ignore_label_, ignore_label_,
  		top_data + n * top_dim_);
  }
  // const Dtype* okok = bottom[0]->cpu_data();
  // for (int i=0; i < 12; ++i)
  //   LOG(ERROR) << "bottom_data[" << i << "] = " << okok[i];
  // const Dtype* okok = top[0]->cpu_data();
  // for (int i=0; i < top[0]->count(); ++i)
  //   LOG(ERROR) << "top_data[" << i << "] = " << okok[i];
  // const Dtype* okok = diff_.cpu_data();
  // for (int i=0; i < diff_.count(); ++i)
  //   LOG(ERROR) << "diff_data[" << i << "] = " << okok[i];
  // LOG(ERROR) << "done forward!!" << std::endl;

}

template <typename Dtype>
void Im2parityLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}


INSTANTIATE_LAYER_GPU_FUNCS(Im2parityLayer);

}  // namespace caffe

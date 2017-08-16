#include <vector>

#include "caffe/layers/replicate_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ReplicateLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const ReplicateParameter& replicate_param = this->layer_param_.replicate_param();
  replicate_axis_ = bottom[0]->CanonicalAxisIndex(replicate_param.axis());
  if (replicate_param.has_num_copies()) {
    num_copies_ = static_cast<int>(replicate_param.num_copies());
    // Don't allow negative indexing for num_copies, a uint32 -- almost
    // certainly unintended.
    CHECK_GE(num_copies_, 1) << "casting num_copies from uint32 to int32 "
			     << "produced negative result; num_copies must satisfy "
			     << "1 <= num_copies";
  } else {
    num_copies_ = 2; // by default
  }
}

template <typename Dtype>
void ReplicateLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Initialize with the first blob.
  vector<int> top_shape = bottom[0]->shape();
  num_on_axis_ = bottom[0]->count(0, replicate_axis_);
  input_size_ = bottom[0]->count(replicate_axis_ + 1);
  for (int i = 1; i < num_copies_; ++i) {
    top_shape[replicate_axis_] += bottom[0]->shape(replicate_axis_);
  }
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void ReplicateLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_cpu_data();
  int offset_dim = 0;
  const int top_dim = top[0]->shape(replicate_axis_);
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const int bottom_dim = bottom[0]->shape(replicate_axis_);
  for (int i = 0; i < num_copies_; ++i) {
    for (int n = 0; n < num_on_axis_; ++n) {
      caffe_copy(bottom_dim * input_size_,
          bottom_data + n * bottom_dim * input_size_,
          top_data + (n * top_dim + offset_dim)
              * input_size_);
    }
    offset_dim += bottom_dim;
  }
}

template <typename Dtype>
void ReplicateLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  int offset_dim = 0;
  const int top_dim = top[0]->shape(replicate_axis_);
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const int bottom_dim = bottom[0]->shape(replicate_axis_);
  caffe_set(bottom_dim*input_size_, Dtype(0), bottom_diff);
  for (int i = 0; i < num_copies_; ++i) {
    if (!propagate_down[i]) { continue; }
    for (int n = 0; n < num_on_axis_; ++n) {
      caffe_add(bottom_dim * input_size_, top_diff +
		(n * top_dim + offset_dim) * input_size_,
		bottom_diff + n * bottom_dim * input_size_, 
		bottom_diff + n * bottom_dim * input_size_);
    }
    offset_dim += bottom_dim;
  }
}

#ifdef CPU_ONLY
STUB_GPU(ReplicateLayer);
#endif

INSTANTIATE_CLASS(ReplicateLayer);
REGISTER_LAYER_CLASS(Replicate);

}  // namespace caffe

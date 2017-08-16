#include <vector>

#include "caffe/layers/replicate_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void Replicate(const int nthreads, const Dtype* in_data,
    const bool forward, const int num_concats, const int replicate_size,
    const int top_dim, const int bottom_dim,
    const int offset_dim, Dtype* out_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int total_replicate_size = replicate_size * bottom_dim;
    const int replicate_num = index / total_replicate_size;
    const int replicate_index = index % total_replicate_size;
    const int top_index = replicate_index +
        (replicate_num * top_dim + offset_dim) * replicate_size;
    if (forward) {
      out_data[top_index] = in_data[index];
    } else {
      // we need to increment to accumlate the diffs
      out_data[index] += in_data[top_index];
    }
  }
}

template <typename Dtype>
void ReplicateLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_gpu_data();
  int offset_dim = 0;
  const int top_dim = top[0]->shape(replicate_axis_);
  const bool kForward = true;
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const int bottom_dim = bottom[0]->shape(replicate_axis_);
  const int bottom_replicate_size = bottom_dim * input_size_;
  const int nthreads = bottom_replicate_size * num_on_axis_;
  for (int i = 0; i < num_copies_; ++i) {
    Replicate<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
        nthreads, bottom_data, kForward, num_on_axis_, input_size_,
        top_dim, bottom_dim, offset_dim, top_data);
    offset_dim += bottom_dim;
  }
}

template <typename Dtype>
void ReplicateLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  int offset_dim = 0;
  const int top_dim = top[0]->shape(replicate_axis_);
  const bool kForward = false;
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int bottom_dim = bottom[0]->shape(replicate_axis_);
  const int bottom_replicate_size = bottom_dim * input_size_;
  const int nthreads = bottom_replicate_size * num_on_axis_;
  caffe_gpu_set(bottom_replicate_size, Dtype(0), bottom_diff);
  for (int i = 0; i < num_copies_; ++i) {
    if (!propagate_down[i]) { continue; }
    Replicate<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
        nthreads, top_diff, kForward, num_on_axis_, input_size_,
        top_dim, bottom_dim, offset_dim, bottom_diff);
    offset_dim += bottom_dim;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ReplicateLayer);

}  // namespace caffe

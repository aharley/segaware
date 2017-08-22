#include <vector>

#include "caffe/layers/dist_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

// void dist_loss_gpu_kernel(const int n, const Dtype* dist_col, const Dtype* parity_col,
template <typename Dtype>
__global__ void dist_loss_gpu_kernel(const int n, const Dtype* dist_col, const Dtype* parity_col,
    const int height_col, const int width_col, const int channels_col, 
    const bool has_ignore_label, const Dtype ignore_label, 
    const Dtype alpha, const Dtype beta, 
    Dtype* diff_col) {
  // for (int index = 0; index < n; ++index) {
  CUDA_KERNEL_LOOP(index, n) {
    int w_out = index % width_col;
    int h_index = index / width_col;
    int h_out = h_index % height_col;
    const Dtype* dist_col_ptr = dist_col;
    const Dtype* parity_col_ptr = parity_col;
    Dtype* diff_col_ptr = diff_col;
    dist_col_ptr += h_out * width_col + w_out;
    parity_col_ptr += h_out * width_col + w_out;
    diff_col_ptr += h_out * width_col + w_out;
    for (int i = 0; i < channels_col; ++i) {
      const Dtype dist =  *dist_col_ptr;
      int parity = *parity_col_ptr;
      if (has_ignore_label && parity==ignore_label) {
	continue;
      } else {
	if (parity)
	  *diff_col_ptr = max(dist-alpha, Dtype(0));
	else
	  *diff_col_ptr = -max(beta-dist, Dtype(0));
      }
      dist_col_ptr += height_col * width_col;
      parity_col_ptr += height_col * width_col;
      diff_col_ptr += height_col * width_col;
    }
  }
}


template <typename Dtype>
void DistLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  int num = bottom[0]->num();
  int height_col = bottom[0]->height();
  int width_col = bottom[0]->width();
  int channels_col = bottom[0]->channels();
  int count = bottom[0]->count();
  // start one kernel per position, then within go through the channels
  int num_kernels = height_col * width_col;

  const Dtype* dist_col = bottom[0]->gpu_data();
  const Dtype* parity_col = bottom[1]->gpu_data();
  Dtype* diff_col = diff_.mutable_gpu_data();
  Dtype loss = 0;
  caffe_gpu_set(height_col * width_col * channels_col, Dtype(0), diff_col);
  // NOLINT_NEXT_LINE(whitespace/operators)
  dist_loss_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                                CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, dist_col, parity_col, 
      height_col, width_col, channels_col, 
      has_ignore_label_, ignore_label_, 
      alpha_, beta_, 
      diff_col);
  CUDA_POST_KERNEL_CHECK;
  caffe_gpu_asum(count,diff_col,&loss);
  const Dtype* dist_cpu = diff_.cpu_data();
  top[0]->mutable_cpu_data()[0] = loss / count;
}

template <typename Dtype>
void DistLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int count = bottom[0]->count();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const Dtype* diff_col = diff_.gpu_data();
  caffe_copy(count, diff_col, bottom_diff);
  caffe_gpu_scal(count, Dtype(1) / count, bottom_diff);

  // Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  // const Dtype* diff_col = diff_.cpu_data();
  // caffe_copy(count, diff_col, bottom_diff);
  // caffe_scal(count, Dtype(1) / num / height_col / width_col, bottom_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(DistLossLayer);

}  // namespace caffe                                                                                                                                                                                       

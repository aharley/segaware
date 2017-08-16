#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/softmax_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SoftmaxLossForwardGPU(const int nthreads,
          const Dtype* prob_data, const Dtype* label, Dtype* loss,
          const int num, const int dim, const int spatial_dim,
          const bool has_ignore_label_, const int ignore_label_,
          Dtype* loss_weights, const Dtype* class_weights_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);
    if (has_ignore_label_ && label_value == ignore_label_) {
      loss[index] = 0;
      loss_weights[index] = 0;
    } else {
      loss[index] = class_weights_data[label_value] * -log(max(prob_data[n * dim + label_value * spatial_dim + s],
                      Dtype(FLT_MIN)));
      loss_weights[index] = class_weights_data[label_value];
    }
  }
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // I think the foward is actually fine, but it's risky.
  Forward_cpu(bottom, top);
  // softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  // const Dtype* prob_data = prob_.gpu_data();
  // const Dtype* label = bottom[1]->gpu_data();
  // const int dim = prob_.count() / outer_num_;
  // const int nthreads = outer_num_ * inner_num_;
  // // Since this memory is not used for anything until it is overwritten
  // // on the backward pass, we use it here to avoid having to allocate new GPU
  // // memory to accumulate intermediate results in the kernel.
  // Dtype* loss_data = bottom[0]->mutable_gpu_diff();
  // // Similarly, this memory is never used elsewhere, and thus we can use it
  // // to avoid having to allocate additional GPU memory.
  // Dtype* loss_weights = prob_.mutable_gpu_diff();
  // caffe_gpu_set(prob_.count(), Dtype(0), loss_weights);
  // const Dtype* class_weights_data = weights_.gpu_data();

  // // NOLINT_NEXT_LINE(whitespace/operators)
  // SoftmaxLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
  //     CAFFE_CUDA_NUM_THREADS>>>(nthreads, prob_data, label, loss_data,
  // 				outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, loss_weights, class_weights_data);
  // Dtype loss;
  // caffe_gpu_asum(nthreads, loss_data, &loss);
  // Dtype batch_weight = -1;
  // // Only launch another CUDA kernel if we actually need the count of valid
  // // outputs.
  // if (normalization_ == LossParameter_NormalizationMode_VALID) {
  //   caffe_gpu_asum(nthreads, loss_weights, &batch_weight);
  // }
  // top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_,
  //                                                       batch_weight);
  // if (top.size() == 2) {
  //   top[1]->ShareData(prob_);
  // }
}

template <typename Dtype>
__global__ void SoftmaxLossBackwardGPU(const int nthreads, const Dtype* top,
          const Dtype* label, Dtype* bottom_diff, const int num, const int dim,
          const int spatial_dim, const bool has_ignore_label_,
           const int ignore_label_, Dtype* loss_weights, const Dtype* class_weights_data) {
  const int channels = dim / spatial_dim;

  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);

    if (has_ignore_label_ && label_value == ignore_label_) {
      for (int c = 0; c < channels; ++c) {
        bottom_diff[n * dim + c * spatial_dim + s] = 0;
      }
      loss_weights[index] = 0;
    } else {
      bottom_diff[n * dim + label_value * spatial_dim + s] -= class_weights_data[label_value];
      loss_weights[index] = class_weights_data[label_value];
    }
  }
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  // It's trickier than it looks!
  Backward_cpu(top, propagate_down, bottom);
  // if (propagate_down[1]) {
  //   LOG(FATAL) << this->type()
  //              << " Layer cannot backpropagate to label inputs.";
  // }
  // if (propagate_down[0]) {
  //   Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  //   const Dtype* prob_data = prob_.gpu_data();
  //   const Dtype* top_data = top[0]->gpu_data();
  //   caffe_gpu_memcpy(prob_.count() * sizeof(Dtype), prob_data, bottom_diff);
  //   const Dtype* label = bottom[1]->gpu_data();
  //   const int dim = prob_.count() / outer_num_;
  //   const int nthreads = outer_num_ * inner_num_;
  //   // Since this memory is never used for anything else,
  //   // we use to to avoid allocating new GPU memory.
  //   Dtype* loss_weights = prob_.mutable_gpu_diff();
  //   caffe_gpu_set(prob_.count(), Dtype(0), loss_weights);
  //   const Dtype* class_weights_data = weights_.gpu_data();
  //   // NOLINT_NEXT_LINE(whitespace/operators)
  //   SoftmaxLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
  //       CAFFE_CUDA_NUM_THREADS>>>(nthreads, top_data, label, bottom_diff,
  // 	  outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, loss_weights, class_weights_data);

  //   LOG(ERROR) << "outer_num_ = " << outer_num_ << ", inner_num_ = " << inner_num_;
  //   Dtype batch_weight = -1;
  //   LOG(ERROR) << "on init, backward batch_weight = " << batch_weight;
  //   // Only launch another CUDA kernel if we actually need the count of valid
  //   // outputs.
  //   if (normalization_ == LossParameter_NormalizationMode_VALID) {
  //     caffe_gpu_asum(nthreads, loss_weights, &batch_weight);
  //   }
  //   LOG(ERROR) << "after asum, batch_weight = " << batch_weight;
  //   const Dtype* lw = prob_.cpu_diff();
  //   const Dtype* cw = weights_.cpu_data();
  //   for (int i=0; i < nthreads; i++) {
  //     LOG(ERROR) << "for example, loss_weights[" << i << "] = " << lw[i];
  //   }
  //   for (int i=0; i < weights_.count(); i++) {
  //     LOG(ERROR) << "also, class_weights[" << i << "] = " << cw[i];
  //   }
  //   const Dtype loss_weight = top[0]->cpu_diff()[0] /
  //                             get_normalizer(normalization_, batch_weight);
  //   caffe_gpu_scal(prob_.count(), loss_weight, bottom_diff);
  // }
}

INSTANTIATE_LAYER_GPU_FUNCS(SoftmaxWithLossLayer);

}  // namespace caffe

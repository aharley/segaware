#include <vector>
#include <cfloat>

// #include "thrust/device_vector.h"
#include "caffe/layers/norm_conv_layer.hpp"
// #include "caffe/util/math_functions.hpp"
// #include "caffe/util/im2dist.hpp"

namespace caffe {

#ifndef CPU_ONLY
template <typename Dtype>
__global__ void kernel_channel_max(const int num, const int channels,
    const int spatial_dim, const Dtype* data, Dtype* out) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    Dtype maxval = -FLT_MAX;
    for (int c = 0; c < channels; ++c) {
      maxval = max(data[(n * channels + c) * spatial_dim + s], maxval);
    }
    out[index] = maxval;
  }
}

template <typename Dtype>
__global__ void kernel_channel_subtract(const int count,
    const int num, const int channels,
    const int spatial_dim, const Dtype* channel_max, Dtype* data) {
  CUDA_KERNEL_LOOP(index, count) {
    int n = index / channels / spatial_dim;
    int s = index % spatial_dim;
    data[index] -= channel_max[n * spatial_dim + s];
  }
}

template <typename Dtype>
__global__ void kernel_exp(const int count, const Dtype* data, Dtype* out) {
  CUDA_KERNEL_LOOP(index, count) {
    out[index] = exp(data[index]);
  }
}

template <typename Dtype>
__global__ void kernel_channel_sum(const int num, const int channels,
    const int spatial_dim, const Dtype* data, Dtype* channel_sum) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    Dtype sum = 0;
    for (int c = 0; c < channels; ++c) {
      sum += data[(n * channels + c) * spatial_dim + s];
    }
    channel_sum[index] = sum;
  }
}

template <typename Dtype>
__global__ void kernel_channel_div(const int count,
    const int num, const int channels,
    const int spatial_dim, const Dtype* channel_sum, Dtype* data) {
  CUDA_KERNEL_LOOP(index, count) {
    int n = index / channels / spatial_dim;
    int s = index % spatial_dim;
    data[index] /= channel_sum[n * spatial_dim + s];
  }
}

template <typename Dtype>
__global__ void kernel_channel_dot(const int num, const int channels,
    const int spatial_dim, const Dtype* data_1, const Dtype* data_2,
    Dtype* channel_dot) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    Dtype dot = 0;
    for (int c = 0; c < channels; ++c) {
      dot += (data_1[(n * channels + c) * spatial_dim + s]
          * data_2[(n * channels + c) * spatial_dim + s]);
    }
    channel_dot[index] = dot;
  }
}

template <typename Dtype>
void NormConvLayer<Dtype>::norm_weight_gpu_gemm(const Dtype* output, Dtype* weights) {
  // prep col_buffer_ and emb_col_buffer_
  const int emb_count = emb_col_buffer_.count();
  // multiply the two
  for (int c=0; c < conv_in_channels_; ++c) {
    caffe_gpu_mul(emb_count,
  	      soft_col_buffer_.gpu_data(),
  	      col_buffer_.gpu_data() + c * emb_count,
  	      res_col_buffer_.mutable_gpu_data() + c * emb_count);
  }
  // gemm into weights
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
  			  kernel_dim_, conv_out_spatial_dim_,
  			  (Dtype)1., output + output_offset_ * g,  
  			  res_col_buffer_.gpu_data() + col_offset_ * g,
  			  (Dtype)1., weights + weight_offset_ * g);
  }
}

template <typename Dtype>
void NormConvLayer<Dtype>::norm_forward_gpu_gemm(const Dtype* weights, 
					      Dtype* output, bool skip_im2col) {
  // prep col_buffer_ and emb_col_buffer_
  const int emb_count = soft_col_buffer_.count();
  // multiply the two
  for (int c=0; c < conv_in_channels_; ++c) {
    caffe_gpu_mul(emb_count,
  	      soft_col_buffer_.gpu_data(),
  	      col_buffer_.gpu_data() + c * emb_count,
  	      res_col_buffer_.mutable_gpu_data() + c * emb_count);
  }
  // gemm into output
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
  			  group_, conv_out_spatial_dim_, kernel_dim_,
  			  (Dtype)1., weights + weight_offset_ * g, 
  			  res_col_buffer_.mutable_gpu_data() + col_offset_ * g,
  			  (Dtype)0., output + output_offset_ * g);
  }
}

template <typename Dtype>
void NormConvLayer<Dtype>::norm_backward_gpu_img_gemm(const Dtype* top_diff,
						      const Dtype* weights, Dtype* input_img) {
  // prep col_buffer_ and emb_col_buffer_
  const int emb_count = emb_col_buffer_.count();
  // gemm into res_col_buffer_
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
  			  conv_out_spatial_dim_, conv_out_channels_ / group_,
  			  (Dtype)1., weights + weight_offset_ * g, top_diff + output_offset_ * g,
  			  (Dtype)0., res_col_buffer_.mutable_gpu_data() + col_offset_ * g);
  }
  // multiply by exp(scale(emb))
  for (int c=0; c < conv_in_channels_; ++c) {
    caffe_gpu_mul(emb_count,
  	      soft_col_buffer_.gpu_data(),
  	      res_col_buffer_.gpu_data() + c * emb_count,
  	      res_col_buffer_.mutable_gpu_data() + c * emb_count);
  }
  // col2im
  if (!is_1x1_ && !bottom_is_im2col_) {
    conv_col2im_gpu(res_col_buffer_.gpu_data(), input_img);
  }
}

template <typename Dtype>
void NormConvLayer<Dtype>::norm_backward_gpu_emb_gemm(const Dtype* top_diff,
						      const Dtype* weights, Dtype* emb_diff) {
  // prep col_buffer_ and emb_col_buffer_
  const int img_count = res_col_buffer_.count();
  const int emb_count = emb_col_buffer_.count();
  // gemm into res_col_buffer_
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
  			  conv_out_spatial_dim_, conv_out_channels_ / group_,
  			  (Dtype)1., weights + weight_offset_ * g, top_diff + output_offset_ * g,
  			  (Dtype)0., res_col_buffer_.mutable_gpu_data() + col_offset_ * g);
  }
  // mult by img
  caffe_gpu_mul(img_count, 
  	    col_buffer_.gpu_data(),
  	    res_col_buffer_.gpu_data(),
  	    res_col_buffer_.mutable_gpu_data());
  // sum down to one channel
  for (int c=1; c < conv_in_channels_; ++c) {
    caffe_gpu_axpy(emb_count,
  	       Dtype(1),
  	       res_col_buffer_.gpu_data() + c * emb_count,
  	       res_col_buffer_.mutable_gpu_data());
  }
  Dtype* sum_data = sum_buffer_.mutable_gpu_data();
  int mask_size = emb_col_buffer_.count(0,channel_axis_);
  // Compute inner1d(top_diff, top_data) and subtract them from the bottom diff.
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_dot<Dtype><<<CAFFE_GET_BLOCKS(1 * conv_out_spatial_dim_),
      CAFFE_CUDA_NUM_THREADS>>>(1, mask_size, conv_out_spatial_dim_,
      res_col_buffer_.cpu_data(), soft_col_buffer_.cpu_data(), sum_data);
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_subtract<Dtype><<<CAFFE_GET_BLOCKS(emb_count),
      CAFFE_CUDA_NUM_THREADS>>>(emb_count, 1, mask_size, conv_out_spatial_dim_,
      sum_data, res_col_buffer_.mutable_gpu_data());
  // elementwise multiplication
  caffe_gpu_mul<Dtype>(emb_count, res_col_buffer_.mutable_gpu_data(), soft_col_buffer_.cpu_data(), res_col_buffer_.mutable_gpu_data());
  // // compute dot(top_diff, top_data) and subtract them from the bottom diff
  // for (int k = 0; k < conv_out_spatial_dim_; ++k) {
  //   sum_data[k] = caffe_cpu_strided_dot<Dtype>(mask_size,
  // 	 res_col_buffer_.cpu_data() + k, conv_out_spatial_dim_,
  // 	 soft_col_buffer_.cpu_data() + k, conv_out_spatial_dim_);
  // }
  // // subtraction
  // caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, mask_size, conv_out_spatial_dim_, 1,
  // 		-1., sum_multiplier_.gpu_data(), sum_data, 1., res_col_buffer_.mutable_gpu_data());
  // // elementwise multiplication
  // caffe_gpu_mul(emb_count, res_col_buffer_.gpu_data(), soft_col_buffer_.gpu_data(), res_col_buffer_.mutable_gpu_data());
  if (scale_term_) {
    // scale the res
    Dtype* scale_factor = this->blobs_[scale_ind_].get()->mutable_cpu_data();
    caffe_gpu_scale(emb_count, 
  		    -scale_factor[0], 
  		    res_col_buffer_.gpu_data(), 
  		    res_col_buffer_.mutable_gpu_data());
  }
  // dist2im
  if (!is_1x1_ && !bottom_is_im2col_) {
    conv_dist2im_gpu(res_col_buffer_.gpu_data(), 
  		     diff_col_buffer_.gpu_data(), 
  		     emb_diff);
  }
}

template <typename Dtype>
void NormConvLayer<Dtype>::backward_gpu_scale(Dtype* scale_diff,
     const Dtype* weights, const Dtype* input_emb, const Dtype* top_diff) {
  // prep col_buffer_ and emb_col_buffer_
  const int img_count = res_col_buffer_.count();
  const int emb_count = emb_col_buffer_.count();
  // gemm into res_col_buffer_
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
  			  conv_out_spatial_dim_, conv_out_channels_ / group_,
  			  (Dtype)1., weights + weight_offset_ * g, top_diff + output_offset_ * g,
  			  (Dtype)0., res_col_buffer_.mutable_gpu_data() + col_offset_ * g);
  }
  // mult by img
  caffe_gpu_mul(img_count, 
  	    col_buffer_.gpu_data(),
  	    res_col_buffer_.gpu_data(),
  	    res_col_buffer_.mutable_gpu_data());
  // sum down to one channel
  for (int c=1; c < conv_in_channels_; ++c) {
    caffe_gpu_axpy(emb_count,
  	       Dtype(1),
  	       res_col_buffer_.gpu_data() + c * emb_count,
  	       res_col_buffer_.mutable_gpu_data());
  }
  Dtype* sum_data = sum_buffer_.mutable_gpu_data();
  int mask_size = emb_col_buffer_.count(0,channel_axis_);
  // Compute inner1d(top_diff, top_data) and subtract them from the bottom diff.
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_dot<Dtype><<<CAFFE_GET_BLOCKS(1 * conv_out_spatial_dim_),
      CAFFE_CUDA_NUM_THREADS>>>(1, mask_size, conv_out_spatial_dim_,
      res_col_buffer_.cpu_data(), soft_col_buffer_.cpu_data(), sum_data);
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_subtract<Dtype><<<CAFFE_GET_BLOCKS(emb_count),
      CAFFE_CUDA_NUM_THREADS>>>(emb_count, 1, mask_size, conv_out_spatial_dim_,
      sum_data, res_col_buffer_.mutable_gpu_data());
  // elementwise multiplication
  caffe_gpu_mul<Dtype>(emb_count, res_col_buffer_.mutable_gpu_data(), soft_col_buffer_.cpu_data(), res_col_buffer_.mutable_gpu_data());
  // // compute dot(top_diff, top_data) and subtract them from the bottom diff
  // for (int k = 0; k < conv_out_spatial_dim_; ++k) {
  //   sum_data[k] = caffe_cpu_strided_dot<Dtype>(mask_size,
  // 	 res_col_buffer_.cpu_data() + k, conv_out_spatial_dim_,
  // 	 soft_col_buffer_.cpu_data() + k, conv_out_spatial_dim_);
  // }
  // // subtraction
  // caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, mask_size, conv_out_spatial_dim_, 1,
  // 		-1., sum_multiplier_.gpu_data(), sum_data, 1., res_col_buffer_.mutable_gpu_data());
  // // elementwise multiplication
  // caffe_gpu_mul(emb_count, res_col_buffer_.gpu_data(), soft_col_buffer_.gpu_data(), res_col_buffer_.mutable_gpu_data());
  // get a fresh embdist
  conv_im2dist_gpu(input_emb, emb_col_buffer_.mutable_gpu_data(), 
  		   diff_col_buffer_.mutable_gpu_data());
  // mult by embdist
  caffe_gpu_mul(emb_count,
  	    emb_col_buffer_.gpu_data(),
  	    res_col_buffer_.gpu_data(),
  	    res_col_buffer_.mutable_gpu_data());
  // mult by scale sign
  caffe_gpu_scale(emb_count, Dtype(-1), res_col_buffer_.gpu_data(), res_col_buffer_.mutable_gpu_data());
  // add it up
  caffe_gpu_gemv<Dtype>(CblasNoTrans, 1, emb_count, 1.,
  			res_col_buffer_.gpu_data(), sum_multiplier_.gpu_data(), 1., scale_diff);
}

template <typename Dtype>
void NormConvLayer<Dtype>::prep_buffers_gpu(const Dtype* weights, 
	const Dtype* input_img, const Dtype* input_emb) {
  const int emb_count = emb_col_buffer_.count();
  // get fresh copies of these
  conv_im2col_gpu(input_img, col_buffer_.mutable_gpu_data());
  conv_im2dist_gpu(input_emb, emb_col_buffer_.mutable_gpu_data(), 
  		   diff_col_buffer_.mutable_gpu_data());
  // for (int i=47600;i<47610;i++)
  //   LOG(ERROR) << "distemb[" << i << "] = " << emb_col_buffer_.cpu_data()[i];
  // LOG(ERROR) << " <<< emb count = " << emb_col_buffer_.count();
  // // for (int i=0;i<0+62500*13;i+=62500)
  // for (int i=47600;i<47610;i++)
  //   LOG(ERROR) << "distemb[" << i << "] = " << emb_col_buffer_.cpu_data()[i];
  // LOG(ERROR) << " <<< ";
  // scale the embs
  if (scale_term_) {
    Dtype* scale_factor = this->blobs_[scale_ind_].get()->mutable_cpu_data();
    caffe_gpu_scale(emb_count, -scale_factor[0], 
		    emb_col_buffer_.gpu_data(), 
		    emb_col_buffer_.mutable_gpu_data());
  }
  // // for (int i=0;i<0+62500*13;i+=62500)
  // for (int i=47600;i<47610;i++)
  //   LOG(ERROR) << "scalemb[" << i << "] = " << emb_col_buffer_.cpu_data()[i];
  // softmax...
  Dtype* sum_data = sum_buffer_.mutable_gpu_data();
  int mask_size = emb_col_buffer_.count(0,channel_axis_);
  caffe_copy(emb_count, emb_col_buffer_.gpu_data(), soft_col_buffer_.mutable_gpu_data());
  // We need to subtract the max to avoid numerical issues, compute the exp,
  // and then normalize.
  // compute max
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_max<Dtype><<<CAFFE_GET_BLOCKS(1 * conv_out_spatial_dim_),
      CAFFE_CUDA_NUM_THREADS>>>(1, mask_size, conv_out_spatial_dim_, soft_col_buffer_.mutable_gpu_data(),
      sum_data);
  // for (int i=47600;i<47610;i++)
  //   LOG(ERROR) << "sum_data[" << i << "] = " << sum_buffer_.cpu_data()[i];
  // subtract
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_subtract<Dtype><<<CAFFE_GET_BLOCKS(emb_count),
      CAFFE_CUDA_NUM_THREADS>>>(emb_count, 1, mask_size, conv_out_spatial_dim_,
      sum_data, soft_col_buffer_.mutable_gpu_data());
  // for (int i=47600;i<47610;i++)
  //   LOG(ERROR) << "subemb[" << i << "] = " << soft_col_buffer_.cpu_data()[i];
  // exponentiate
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_exp<Dtype><<<CAFFE_GET_BLOCKS(emb_count), CAFFE_CUDA_NUM_THREADS>>>(
      emb_count, soft_col_buffer_.mutable_gpu_data(), soft_col_buffer_.mutable_gpu_data());
  // for (int i=47600;i<47610;i++)
  //   LOG(ERROR) << "expemb[" << i << "] = " << soft_col_buffer_.cpu_data()[i];
  // sum after exp
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_sum<Dtype><<<CAFFE_GET_BLOCKS(1 * conv_out_spatial_dim_),
      CAFFE_CUDA_NUM_THREADS>>>(1, mask_size, conv_out_spatial_dim_, soft_col_buffer_.mutable_gpu_data(),
      sum_data);
  // for (int i=47600;i<47610;i++)
  //   LOG(ERROR) << "sum_data[" << i << "] = " << sum_buffer_.cpu_data()[i];
  // divide
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_div<Dtype><<<CAFFE_GET_BLOCKS(emb_count),
      CAFFE_CUDA_NUM_THREADS>>>(emb_count, 1, mask_size, conv_out_spatial_dim_,
      sum_data, soft_col_buffer_.mutable_gpu_data());
  // for (int i=47600;i<47610;i++)
  //   LOG(ERROR) << "divemb[" << i << "] = " << soft_col_buffer_.cpu_data()[i];
}

// template <typename Dtype>
// void NormConvLayer<Dtype>::norm_backward_gpu_all(const Dtype* top_diff,
//   const Dtype* weights, const Dtype* input_img, const Dtype* input_emb, 
//   Dtype* weight_diff, Dtype* img_diff, Dtype* emb_diff, Dtype* scale_diff) {
//   // doesn't work yet
// }

template <typename Dtype>
void NormConvLayer<Dtype>::forward_gpu_bias(Dtype* output,
					    const Dtype* bias) {
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
			out_spatial_dim_, 1, (Dtype)1., bias, sum_multiplier_.gpu_data(),
			(Dtype)1., output);
}

template <typename Dtype>
void NormConvLayer<Dtype>::backward_gpu_bias(Dtype* bias,
					     const Dtype* input) {
  caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1.,
			input, sum_multiplier_.gpu_data(), 1., bias);
}

template <typename Dtype>
void NormConvLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
				      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  const Dtype* bottom_img = bottom[0]->gpu_data();
  const Dtype* bottom_emb = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  for (int n = 0; n < num_; ++n) {
    prep_buffers_gpu(weight, 
		     bottom_img + n * bottom_dim_,
		     bottom_emb + n * emb_bottom_dim_);
    norm_forward_gpu_gemm(weight,
			  top_data + n * top_dim_);
    if (bias_term_) {
      const Dtype* bias = this->blobs_[1]->gpu_data();
      forward_gpu_bias(top_data + n * top_dim_, bias);
    }
  }
}

template <typename Dtype>
void NormConvLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* bottom_img = bottom[0]->gpu_data();
  const Dtype* bottom_emb = bottom[1]->gpu_data();
  Dtype* bottom_img_diff = bottom[0]->mutable_gpu_diff();
  Dtype* bottom_emb_diff = bottom[1]->mutable_gpu_diff();

  // Bias gradient, if necessary.
  if (bias_term_ && this->param_propagate_down_[1]) {
    Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
    for (int n = 0; n < num_; ++n) {
      backward_gpu_bias(bias_diff, top_diff + n * top_dim_);
    }
  }
  if (this->param_propagate_down_[0] 
      || (scale_term_ && this->param_propagate_down_[scale_ind_])
      || propagate_down[0] || propagate_down[1]) {
    for (int n = 0; n < num_; ++n) {
      // commonly we will want to bprop to everything: weights, image, embeddings, and scale.
      // we can save a bit of time doing these together.
      // if (param_propagate_down_[0] && scale_term_ && 
      // 	  param_propagate_down_[scale_ind_] &&
      // 	  propagate_down[0] && propagate_down[1]) {
      // 	Dtype* scale_diff = blobs_[scale_ind_]->mutable_cpu_diff();
      // 	norm_backward_gpu_all(top_diff + n * top_dim_, 
      // 					  weight,
      // 					  bottom_img + n * bottom_dim_,
      // 					  bottom_emb + n * emb_bottom_dim_,
      // 					  weight_diff,
      // 					  bottom_img_diff + n * bottom_dim_,
      // 					  bottom_emb_diff + n * emb_bottom_dim_,
      // 					  scale_diff);
      // } else {
      // all except scale need a fresh run of im2col and im2dist for data "n"
      if (this->param_propagate_down_[0] || propagate_down[0] || propagate_down[1])
	prep_buffers_gpu(weight, 
			 bottom_img + n * bottom_dim_,
			 bottom_emb + n * emb_bottom_dim_);
      // gradient w.r.t. weight. Note that we will accumulate diffs.
      if (this->param_propagate_down_[0])
	norm_weight_gpu_gemm(top_diff + n * top_dim_, weight_diff);
      // gradient w.r.t. bottom data, if necessary.
      if (propagate_down[0])
	norm_backward_gpu_img_gemm(top_diff + n * top_dim_, weight,
				   bottom_img_diff + n * bottom_dim_);
      if (propagate_down[1])
	norm_backward_gpu_emb_gemm(top_diff + n * top_dim_, weight,
				   bottom_emb_diff + n * emb_bottom_dim_);
      // gradient w.r.t. scale, if necessary
      if (scale_term_ && this->param_propagate_down_[scale_ind_]) {
	Dtype* scale_diff = this->blobs_[scale_ind_]->mutable_gpu_diff();
	backward_gpu_scale(scale_diff, weight,
			   bottom_emb + n * emb_bottom_dim_,
			   top_diff + n * top_dim_);
      }
    }
    // }
  }
}
#endif

INSTANTIATE_LAYER_GPU_FUNCS(NormConvLayer);

}  // namespace caffe

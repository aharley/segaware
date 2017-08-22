#ifndef CAFFE_AUGMENTATION_LAYER_HPP_
#define CAFFE_AUGMENTATION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class AugmentationLayer : public Layer<Dtype> {
 public:
  explicit AugmentationLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Augmentation"; }
  virtual inline int ExactNumBottomBlobs() const { return 3; }
  virtual inline int ExactNumTopBlobs() const { return 4; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  virtual void Translation(float t, cv::Mat im, Dtype* im_final, int w, int h, int channels, int num);
  virtual void TranslationUV(float t, cv::Mat im, Dtype* im_final, Dtype* mask, int w, int h, int channels, int num);
  virtual void Rotation(float angle, cv::Mat im, Dtype* im_final, int w, int h, int channels, int num);
  virtual void RotateUV(float angle, cv::Mat im, Dtype* im_final, Dtype* mask, int w, int h, int channels, int num);
  virtual void Scale(float scale, cv::Mat im, Dtype* im_final, int w, int h, int channels, int num, int sW, int sH);
  virtual void ScaleUV(float scale, cv::Mat im, Dtype* im_final, Dtype* mask, int w, int h, int channels, int num, int sW, int sH);
  virtual cv::Mat blob2Mat(const Dtype* im, int w, int h, int channels, int num);

 private:
   int num_transformations_;
   float max_translation_, min_translation_;
   float max_rotation_, min_rotation_;
   float max_scaling_, min_scaling_;
   float max_mcc_, min_mcc_;
   float max_gamma_, min_gamma_;
   float max_gaussian_, min_gaussian_;
   float max_contrast_, min_contrast_;
   float max_brightness_, min_brightness_;
};
} // namespace caffe

#endif // CAFFE_AUGMENTATION_LAYER_HPP

#include <vector>
#include <math.h>

#include "caffe/layers/augmentation_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define PI 3.14159265

namespace caffe {

template <typename Dtype>
void AugmentationLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){
  if (bottom.size() != 3)
    {
      LOG(INFO)<<"Number of bottom blobs must be 3";
    }
  else{
    num_transformations_ = this->layer_param_.aug_param().num_transformations();
    max_translation_ = this->layer_param_.aug_param().max_translation();
    min_translation_ = this->layer_param_.aug_param().min_translation();
    max_rotation_ = this->layer_param_.aug_param().max_rotation();
    min_rotation_ = this->layer_param_.aug_param().min_rotation();
    max_scaling_ = this->layer_param_.aug_param().max_scaling();
    min_scaling_ = this->layer_param_.aug_param().min_scaling();
    max_mcc_ = this->layer_param_.aug_param().max_mcc();
    min_mcc_ = this->layer_param_.aug_param().min_mcc();
    max_gamma_ = this->layer_param_.aug_param().max_gamma();
    min_gamma_ = this->layer_param_.aug_param().min_gamma();
    max_gaussian_ = this->layer_param_.aug_param().max_gaussian();
    min_gaussian_ = this->layer_param_.aug_param().min_gaussian();
    max_contrast_ = this->layer_param_.aug_param().max_contrast();
    min_contrast_ = this->layer_param_.aug_param().min_contrast();
    max_brightness_ = this->layer_param_.aug_param().max_brightness();
    min_brightness_ = this->layer_param_.aug_param().min_brightness(); 
  }
}

template <typename Dtype>
void AugmentationLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){
  top[0]->Reshape(bottom[0]->shape());
  top[1]->Reshape(bottom[1]->shape());
  top[2]->Reshape(bottom[2]->shape());
  top[3]->Reshape(bottom[2]->num(), bottom[2]->channels()/2, bottom[2]->height(), bottom[2]->width());
}

template <typename Dtype>
cv::Mat AugmentationLayer<Dtype>::blob2Mat(const Dtype* im, int w, int h, int channels, int num){
  if (channels==3){
    cv::Mat ans = cv::Mat(cv::Size(w,h), CV_32FC3);
    for (int x = 0; x < ans.rows; ++x){
      for (int y = 0; y < ans.cols; ++y){
	ans.at<cv::Vec3f>(x,y).val[0] = im[(((num)*channels+2)*h+x)*w+y];
	ans.at<cv::Vec3f>(x,y).val[1] = im[(((num)*channels+1)*h+x)*w+y];
	ans.at<cv::Vec3f>(x,y).val[2] = im[(((num)*channels+0)*h+x)*w+y];
      }
    }
    return ans;
  }
  else{
    cv::Mat ans = cv::Mat(cv::Size(w,h), CV_32FC2);
    for (int x = 0; x < ans.rows; ++x){
      for (int y = 0; y < ans.cols; ++y){
	ans.at<cv::Vec2f>(x,y).val[0] = im[(((num)*channels+0)*h+x)*w+y];
	ans.at<cv::Vec2f>(x,y).val[1] = im[(((num)*channels+1)*h+x)*w+y];
      }
    }
    return ans;
  }
}

template <typename Dtype>
void AugmentationLayer<Dtype>::Translation(float t, cv::Mat im, Dtype* im_final, int w, int h, int channels, int num){
  cv::Mat dst_f;	
  if(t < 0){
    cv::Rect ROI(0,0,w+(int)t,h+(int)t);
    cv::Mat temp_dst = im(ROI);
    copyMakeBorder(temp_dst, dst_f, -1*t, 0, -1*t, 0, cv::BORDER_CONSTANT, 0);
  }
  else{
    cv::Rect ROI((int)t,(int)t,w-(int)t,h-(int)t);
    cv::Mat temp_dst = im(ROI);
    copyMakeBorder(temp_dst, dst_f, 0, t, 0, t, cv::BORDER_CONSTANT, 0);
  }
  for(int x=0;x<dst_f.rows;x++){
    for(int y=0;y<dst_f.cols;y++){
      im_final[(((num)*channels+2)*h+x)*w+y] = dst_f.at<cv::Vec3f>(x,y).val[0];
      im_final[(((num)*channels+1)*h+x)*w+y] = dst_f.at<cv::Vec3f>(x,y).val[1];
      im_final[(((num)*channels+0)*h+x)*w+y] = dst_f.at<cv::Vec3f>(x,y).val[2];
    }
  }
}

//Translate Flo
template <typename Dtype>
void AugmentationLayer<Dtype>::TranslationUV(float t, cv::Mat im, Dtype* im_final, Dtype* mask, int w, int h, int channels, int num){
  cv::Mat dst, dst_f;
  if(t < 0){
    cv::Rect ROI(0,0,w+(int)t,h+(int)t);
    cv::Mat temp_dst = im(ROI);
    copyMakeBorder(temp_dst, dst_f, -1*t, 0, -1*t, 0, cv::BORDER_CONSTANT, 1);
    copyMakeBorder(temp_dst, dst, -1*t, 0, -1*t, 0, cv::BORDER_CONSTANT, 0);
  }
  else{
    cv::Rect ROI((int)t,(int)t,w-(int)t,h-(int)t);
    cv::Mat temp_dst = im(ROI);
    copyMakeBorder(temp_dst, dst_f, 0, t, 0, t, cv::BORDER_CONSTANT, 1);
    copyMakeBorder(temp_dst, dst, 0, t, 0, t, cv::BORDER_CONSTANT, 0);
  }
  //put dst in the im_final
  for(int x=0;x<dst_f.rows;x++){
    for(int y=0;y<dst_f.cols;y++){
      im_final[(((num)*channels+0)*h+x)*w+y] = dst_f.at<cv::Vec2f>(x,y).val[0];
      im_final[(((num)*channels+1)*h+x)*w+y] = dst_f.at<cv::Vec2f>(x,y).val[1];
      if(t<0){
	im_final[(((num)*channels+0)*h+x)*w+y] = im_final[(((num)*channels+0)*h+x)*w+y]*((float)w/((float)w+t));
	im_final[(((num)*channels+1)*h+x)*w+y]= im_final[(((num)*channels+1)*h+x)*w+y]*((float)h/((float)h+t));	
      }
      else{
	im_final[(((num)*channels+0)*h+x)*w+y] = im_final[(((num)*channels+0)*h+x)*w+y]*((float)w/((float)w-t));
	im_final[(((num)*channels+1)*h+x)*w+y]= im_final[(((num)*channels+1)*h+x)*w+y]*((float)h/((float)h-t));
      }
      mask[((num)*h+x)*w+y] = (dst_f.at<cv::Vec2f>(x,y).val[0] - dst.at<cv::Vec2f>(x,y).val[0]);
    }
  }
}

//Rotate Image
template <typename Dtype>
void AugmentationLayer<Dtype>::Rotation(float angle, cv::Mat im, Dtype* im_final, int w, int h, int channels, int num){
  cv::Mat dst, dst_f;
  //Centre of image
  cv::Point2f pt(im.rows/2., im.cols/2.);    
  cv::Mat r = getRotationMatrix2D(pt, (double)angle, 1.0);
  warpAffine(im, dst, r, cv::Size(w, h),cv::INTER_LINEAR,cv::BORDER_CONSTANT,cv::Scalar(0,0,0));
  //put it back in Dtype*
  for(int x=0;x<dst.rows;x++){
    for(int y=0;y<dst.cols;y++){
      im_final[(((num)*channels+2)*h+x)*w+y] = dst.at<cv::Vec3f>(x,y).val[0];
      im_final[(((num)*channels+1)*h+x)*w+y] = dst.at<cv::Vec3f>(x,y).val[1];
      im_final[(((num)*channels+0)*h+x)*w+y] = dst.at<cv::Vec3f>(x,y).val[2];
    }
  }
}

//Rotate Flo
template <typename Dtype>
void AugmentationLayer<Dtype>::RotateUV(float angle, cv::Mat im, Dtype* im_final, Dtype* mask, int w, int h, int channels, int num){
  cv::Mat dst, dst_f;
  float temp, temp_2;
  //Centre of image
  cv::Point2f pt(im.rows/2., im.cols/2.);
  cv::Mat r = getRotationMatrix2D(pt, (double)angle/*PI/180.*/, 1.0);
  warpAffine(im, dst, r, cv::Size(w, h),cv::INTER_LINEAR,cv::BORDER_CONSTANT,cv::Scalar(0,0));
  warpAffine(im, dst_f, r, cv::Size(w, h),cv::INTER_LINEAR,cv::BORDER_CONSTANT,cv::Scalar(1,1));
  //put it back in Dtype*
  for(int x=0;x<dst_f.rows;x++){
    for(int y=0;y<dst_f.cols;y++){
      temp = dst_f.at<cv::Vec2f>(x,y).val[0];
      temp_2 = dst_f.at<cv::Vec2f>(x,y).val[1];
      //Rotate each element
      /**/
      im_final[(((num)*channels+0)*h+x)*w+y] = ((temp*cos((angle)*PI/180)) - temp_2*sin((angle)*PI/180)); //U
      im_final[(((num)*channels+1)*h+x)*w+y] = ((temp*sin((angle)*PI/180)) + temp_2*cos((angle)*PI/180)); //V
      mask[((num)*h+x)*w+y] = (dst_f.at<cv::Vec2f>(x,y).val[0] - dst.at<cv::Vec2f>(x,y).val[0]);
    }
  }
}

//Scale image
template <typename Dtype>
void AugmentationLayer<Dtype>::Scale(float scale, cv::Mat im, Dtype* im_final, int w, int h, int channels, int num, int sW, int sH){
  cv::Mat dst;
  resize(im,dst,cv::Size(),scale,scale);
  if(scale>1.0){
    cv::Rect ROI(sW,sH,w,h);
    //cv::Rect ROI(0,0,w,h);
    cv::Mat temp_dst = dst(ROI);
    for(int x=0;x<temp_dst.rows;x++){
      for(int y=0;y<temp_dst.cols;y++){
	im_final[(((num)*channels+2)*h+x)*w+y] = temp_dst.at<cv::Vec3f>(x,y).val[0];
	im_final[(((num)*channels+1)*h+x)*w+y] = temp_dst.at<cv::Vec3f>(x,y).val[1];
	im_final[(((num)*channels+0)*h+x)*w+y] = temp_dst.at<cv::Vec3f>(x,y).val[2];
      }
    }
  }
  else{
    int lr = floor((w - dst.cols)/2);
    int ud = floor((h - dst.rows)/2);
    cv::Mat temp_dst;
    copyMakeBorder(dst, temp_dst, ud, ud + (int)(h - dst.rows)%2, lr, lr + (int)(w - dst.cols)%2, cv::BORDER_CONSTANT, 0);
    for(int x=0;x<temp_dst.rows;x++){
      for(int y=0;y<temp_dst.cols;y++){
	im_final[(((num)*channels+2)*h+x)*w+y] = temp_dst.at<cv::Vec3f>(x,y).val[0];
	im_final[(((num)*channels+1)*h+x)*w+y] = temp_dst.at<cv::Vec3f>(x,y).val[1];
	im_final[(((num)*channels+0)*h+x)*w+y] = temp_dst.at<cv::Vec3f>(x,y).val[2];
      }
    }
  }
}

//Scale Flo
template <typename Dtype>
void AugmentationLayer<Dtype>::ScaleUV(float scale, cv::Mat im, Dtype* im_final, Dtype* mask, int w, int h, int channels, int num, int sW, int sH){
  cv::Mat dst;
  resize(im,dst,cv::Size(),scale,scale);
  //int relative_scale = (scale+(scale/10.))/scale;
  //Multiply each element by scale
  if(scale>1.0){
    cv::Rect ROI(sW,sH,w,h);
    //cv::Rect ROI(0,0,w,h);
    cv::Mat temp_dst = dst(ROI);
    for(int x=0;x<temp_dst.rows;x++){
      for(int y=0;y<temp_dst.cols;y++){
	im_final[(((num)*channels+0)*h+x)*w+y] = temp_dst.at<cv::Vec2f>(x,y).val[0];
	im_final[(((num)*channels+1)*h+x)*w+y] = temp_dst.at<cv::Vec2f>(x,y).val[1];
	im_final[(((num)*channels+0)*h+x)*w+y] = im_final[(((num)*channels+0)*h+x)*w+y]*scale;//*relative_scale;
	im_final[(((num)*channels+1)*h+x)*w+y]= im_final[(((num)*channels+1)*h+x)*w+y]*scale;//relative_scale;
      }
    }
  }
  else{
    int lr = floor((w - dst.cols)/2);
    int ud = floor((h - dst.rows)/2);
    float global_scale = (1./scale);
    cv::Mat dst_f, temp_dst;
    copyMakeBorder(dst, temp_dst, ud, ud + (int)(h - dst.rows)%2, lr, lr + (int)(w - dst.cols)%2, cv::BORDER_CONSTANT, 0);
    copyMakeBorder(dst, dst_f, ud, ud + (int)(h - dst.rows)%2, lr, lr + (int)(w - dst.cols)%2, cv::BORDER_CONSTANT, 1);
    for(int x=0;x<dst_f.rows;x++){
      for(int y=0;y<dst_f.cols;y++){
	im_final[(((num)*channels+0)*h+x)*w+y] = dst_f.at<cv::Vec2f>(x,y).val[0];
	im_final[(((num)*channels+1)*h+x)*w+y] = dst_f.at<cv::Vec2f>(x,y).val[1];
	im_final[(((num)*channels+0)*h+x)*w+y] = im_final[(((num)*channels+0)*h+x)*w+y]*global_scale;//*relative_scale;
	im_final[(((num)*channels+1)*h+x)*w+y]= im_final[(((num)*channels+1)*h+x)*w+y]*global_scale;//*relative_scale;
	mask[((num)*h+x)*w+y] = (dst_f.at<cv::Vec2f>(x,y).val[0] - temp_dst.at<cv::Vec2f>(x,y).val[0]);
      }
    }
  }
}

template <typename Dtype>
void AugmentationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  //Img0, Img1, Flo
  const Dtype* im0 = bottom[0]->cpu_data();
  const Dtype* im1 = bottom[1]->cpu_data();
  const Dtype* flow = bottom[2]->cpu_data();
  Dtype* img0_final = top[0]->mutable_cpu_data();
  Dtype* img1_final = top[1]->mutable_cpu_data();
  Dtype* flo_final = top[2]->mutable_cpu_data();
  Dtype* mask_final = top[3]->mutable_cpu_data();

  const int count = bottom[0]->count();
  const int batch = bottom[0]->num();
  const int h = bottom[0]->height();
  const int w = bottom[0]->width();
  const int channels = bottom[0]->channels();
  const int flo_channels = bottom[2]->channels();

  // caffe_set(top[0]->count(), Dtype(0), img0_final);
  // caffe_set(top[1]->count(), Dtype(0), img1_final);
  // caffe_set(top[2]->count(), Dtype(0), flo_final);
  // caffe_set(top[3]->count(), Dtype(0), mask_final);

  //Dtype* normal = (Dtype*) malloc(channels*w*h*sizeof(Dtype));
  Dtype* normal_1D = (Dtype*) malloc(w*h*sizeof(Dtype));
  int option, x, y, z;
  float t, c, translation_param;
  cv::Mat img0, img1, flo;
  //Loop through the number of transformations
  for(int j=0;j<batch;j++){
    for(int i=0;i<num_transformations_;i++){
      if(i==0){
	img0 = blob2Mat(im0,w,h,channels,j);
	img1 = blob2Mat(im1,w,h,channels,j);
	flo = blob2Mat(flow,w,h,flo_channels,j);
	// option = rand() % 3 + 1;
      }
      else{
	img0 = blob2Mat(img0_final,w,h,channels,j);
	img1 = blob2Mat(img1_final,w,h,channels,j);
	flo = blob2Mat(flo_final,w,h,flo_channels,j);
	// option = rand() % 5 + 4;
      }
      option = i+1; // num_transformations had better be 8
			
      //option = 4;
      switch(option){
	//Translation [-0.2,0.2]
      case 1:
	{
	  t = min_translation_ + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(max_translation_ - min_translation_)));
	  translation_param = (int) (t * w);
	  // printf("Transformation %d for Image Pair %d: Translation by %f pixels\n", i+1, j+1, translation_param);
	  Translation(translation_param, img0 , img0_final, w, h, channels, j);
	  // Translation(translation_param+(translation_param/(float)10.), img1, img1_final, w, h, channels, j);
	  Translation(translation_param, img1, img1_final, w, h, channels, j);
	  TranslationUV(translation_param, flo, flo_final, mask_final, w, h, flo_channels, j);
	  break;
	}
	//Rotation [-17,17]
      case 2:
	{  
	  // the mask does not line up here
	  // /**/
	  // t = min_rotation_ + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(max_rotation_ - min_rotation_)));
	  // // printf("Transformation %d for Image Pair %d: Rotation by %f degrees\n", i+1, j+1, t);
	  // if(t<0.0){t = t + 360.0;}
	  // //printf("Angle = %f\n", t);
	  // Rotation(t, img0, img0_final, w, h, channels, j);
	  // Rotation(t, img1, img1_final, w, h, channels, j);
	  // RotateUV(t, flo, flo_final,mask_final, w, h, flo_channels, j);
	  break;
	}
	//Scaling [0.9, 2.0]
      case 3:
	{  
	  // the mask does not line up here
	  // t = min_scaling_ + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(max_scaling_ - min_scaling_)));
	  // // t = 1.0 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(2.0 - 1.0)));
	  // if(t>1.0){
	  //   if(abs(t - min_scaling_) < 0.01){
	  //     t += 0.01;
	  //   }
	  //   // printf("Transformation %d for Image Pair %d: Scaling by %f\n", i+1, j+1, t);
	  //   int temp = t*w;
	  //   int sW = rand() % (temp - w) + 0;
	  //   temp = t*h;
	  //   int sH = rand() % (temp - h) + 0;
	  //   Scale(t, img0, img0_final, w, h, channels, j, sW, sH);
	  //   Scale(t, img1, img1_final, w, h, channels, j, sW, sH);
	  //   ScaleUV(t, flo, flo_final, mask_final, w, h, flo_channels, j, sW, sH);
	  // }
	  // else{
	  //   // printf("Transformation %d for Image Pair %d: Scaling by %f\n", i+1, j+1, t);
	  //   int sW = 0;
	  //   int sH = 0;
	  //   Scale(t, img0, img0_final, w, h, channels,j, sW, sH);
	  //   Scale(t, img1, img1_final, w, h, channels,j, sW, sH);
	  //   ScaleUV(t, flo, flo_final, mask_final, w, h, flo_channels,j, sW, sH);
	  // }
	  break;
	}
	//Gaussian Noise [0, 0.04]
      case 4:
	{
	  t = min_gaussian_ + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(max_gaussian_ - min_gaussian_)));
	  // printf("Transformation %d for Image Pair %d: Gaussian Noise by %f\n", i+1, j+1, t);
	  caffe_rng_gaussian(count/(channels * batch), (Dtype)0.0, (Dtype)t, normal_1D);
	  for(x=0;x<h;x++){
	    for(y=0;y<w;y++){
	      for(z=0;z<channels;z++){
	  	img0_final[(((j)*channels+z)*h+x)*w+y] = img0.at<cv::Vec3f>(x,y).val[2-z] + normal_1D[x*w+y];
	  	img1_final[(((j)*channels+z)*h+x)*w+y] = img1.at<cv::Vec3f>(x,y).val[2-z] + normal_1D[x*w+y];
	      }
	      flo_final[(((j)*flo_channels+0)*h+x)*w+y] = flo.at<cv::Vec2f>(x,y).val[0];
	      flo_final[(((j)*flo_channels+1)*h+x)*w+y] = flo.at<cv::Vec2f>(x,y).val[1];
	    }
	    t = min_gaussian_ + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(max_gaussian_ - min_gaussian_)));
	    caffe_rng_gaussian(count/(channels * batch), (Dtype)0.0, (Dtype)t, normal_1D);
	  }
	  break;
	}
	//Multiplicative Color Changes [0.5, 2]
      case 5:
	{
	  c = min_mcc_ + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(max_mcc_ - min_mcc_)));
	  // printf("Transformation %d for Image Pair %d: Multiplicative Color Changes by %f \n", i+1, j+1, c);
	  for(z=0;z<channels;z++){
	    for(x=0;x<h;x++){
	      for(y=0;y<w;y++){
	  	img0_final[(((j)*channels+z)*h+x)*w+y] = img0.at<cv::Vec3f>(x,y).val[2-z] * c;
	  	img1_final[(((j)*channels+z)*h+x)*w+y] = img1.at<cv::Vec3f>(x,y).val[2-z] * c;
		flo_final[(((j)*flo_channels+0)*h+x)*w+y] = flo.at<cv::Vec2f>(x,y).val[0];
		flo_final[(((j)*flo_channels+1)*h+x)*w+y] = flo.at<cv::Vec2f>(x,y).val[1];
	      }
	    }
	    c = min_mcc_ + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(max_mcc_ - min_mcc_)));
	  }
	  break;
	}
	//Contrast [-0.8,0.4]
      case 6:
	{  
	  t = min_contrast_ + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(max_contrast_ - min_contrast_)));
	  // printf("Transformation %d for Image Pair %d: Contrast by %f\n", i+1, j+1, t);
	  for(x=0;x<h;x++){
	    for(y=0;y<w;y++){
	      for(z=0;z<channels;z++){
		img0_final[(((j)*channels+z)*h+x)*w+y] = ((t * (img0.at<cv::Vec3f>(x,y).val[2-z] - 0.5)) + 0.5);
		img1_final[(((j)*channels+z)*h+x)*w+y] = ((t * (img1.at<cv::Vec3f>(x,y).val[2-z] - 0.5)) + 0.5);
		flo_final[(((j)*flo_channels+0)*h+x)*w+y] = flo.at<cv::Vec2f>(x,y).val[0];
		flo_final[(((j)*flo_channels+1)*h+x)*w+y] = flo.at<cv::Vec2f>(x,y).val[1];
	      }
	    }
	  }
	  break;
	}
	//Gamma Values [0.7,1.5]
      case 7:
	{  
	  t = min_gamma_ + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(max_gamma_ - min_gamma_)));
	  //t = 1.0/(min_gamma_ + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(max_gamma_ - min_gamma_))));
	  // printf("Transformation %d for Image Pair %d: Gamma by %f\n", i+1, j+1, t);
	  for(x=0;x<h;x++){
	    for(y=0;y<w;y++){
	      for(z=0;z<channels;z++){
	  	img0_final[(((j)*channels+z)*h+x)*w+y] = (pow((double)img0.at<cv::Vec3f>(x,y).val[2-z], (double)t));
	  	img1_final[(((j)*channels+z)*h+x)*w+y] = (pow((double)img1.at<cv::Vec3f>(x,y).val[2-z], (double)t));
	      }
	      flo_final[(((j)*flo_channels+0)*h+x)*w+y] = flo.at<cv::Vec2f>(x,y).val[0];
	      flo_final[(((j)*flo_channels+1)*h+x)*w+y] = flo.at<cv::Vec2f>(x,y).val[1];
	    }
	  }
	  break;
	}
	//Additive Brightness always sigma of 0.2
      case 8:
	{
	  caffe_rng_gaussian(count/(channels * batch), (Dtype)0.0, (Dtype)0.2, normal_1D);
	  // printf("Transformation %d for Image Pair %d: Additive Brightness\n", i+1, j+1);
	  for(x=0;x<h;x++){
	    for(y=0;y<w;y++){
	      for(z=0;z<channels;z++){
	  	img0_final[(((j)*channels+z)*h+x)*w+y] = img0.at<cv::Vec3f>(x,y).val[2-z] + normal_1D[x*w+y];
	  	img1_final[(((j)*channels+z)*h+x)*w+y] = img1.at<cv::Vec3f>(x,y).val[2-z] + normal_1D[x*w+y];
	      }
	      flo_final[(((j)*flo_channels+0)*h+x)*w+y] = flo.at<cv::Vec2f>(x,y).val[0];
	      flo_final[(((j)*flo_channels+1)*h+x)*w+y] = flo.at<cv::Vec2f>(x,y).val[1];
	    }
	  }
	  break;
	}
      default:
	{
	  // ain't do nothing
	}
      } // end switch
    } // end for transformations
    
    // // finally, do some mean sub. 
    // //Mean subtraction img0 and img1
    // for(x=0;x<h;x++){
    //   for(y=0;y<w;y++){
    // 	img0_final[(((j)*channels+0)*h+x)*w+y] -= 0.411451;
    // 	img0_final[(((j)*channels+1)*h+x)*w+y] -= 0.432060;
    // 	img0_final[(((j)*channels+2)*h+x)*w+y] -= 0.450141;
    // 	img1_final[(((j)*channels+0)*h+x)*w+y] -= 0.411451;
    // 	img1_final[(((j)*channels+1)*h+x)*w+y] -= 0.432060;
    // 	img1_final[(((j)*channels+2)*h+x)*w+y] -= 0.450141;
    //   }
    // }
  }
  free(normal_1D);
}

template <typename Dtype>
void AugmentationLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
  // NOTHING
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(AugmentationLayer, Forward);
#endif

INSTANTIATE_CLASS(AugmentationLayer);
REGISTER_LAYER_CLASS(Augmentation);

} // namespace caffe

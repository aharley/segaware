#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/layers/flo_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
bool FloDataLayer<Dtype>::readFloFile(string filename, Dtype* data, int& xSize, int &ySize)
{
    FILE *stream = fopen(filename.c_str(), "rb");
    if (stream == 0)
    {
        LOG(ERROR) << "Could not open or find file " << filename;
        return false;
    }

    try {
      float help;
      if(0 == fread(&help,sizeof(float),1,stream) || help != 202021.25) throw 0;
      if(0 == fread(&xSize,sizeof(int),1,stream)) throw 0;
      if(0 == fread(&ySize,sizeof(int),1,stream)) throw 0;
      
      if(data == NULL) {fclose(stream); return true;}

      //data=new float[xSize*ySize*2];

      for (int y = 0; y < ySize; y++)
          for (int x = 0; x < xSize; x++) {
              if(0 == fread(&data[y*xSize+x],sizeof(float),1,stream)) throw 0;
              if(0 == fread(&data[y*xSize+x+xSize*ySize],sizeof(float),1,stream)) throw 0;
          }
    } catch(int err) {
      fclose(stream);
      LOG(FATAL) << "File corrupted: " << filename;
    }
    fclose(stream);

    return true;
}


template <typename Dtype>
FloDataLayer<Dtype>::~FloDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void FloDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  string root_folder = this->layer_param_.image_data_param().root_folder();

  // Read the file with filenames and labels
  const string& source = this->layer_param_.image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string filename;
  while (infile >> filename) {
    lines_.push_back(std::make_pair(filename, 0));
  }

  if (this->layer_param_.image_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.image_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  
  // Read an image, and use it to initialize the top blob.
  int xSize, ySize;
  CHECK(readFloFile(root_folder + lines_[lines_id_].first, NULL, xSize, ySize))
      << "Could not load " << lines_[lines_id_].first;
  
  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> top_shape = vector<int>(4);
  top_shape[0] = 1;
  top_shape[1] = 2;
  top_shape[2] = ySize;
  top_shape[3] = xSize;
  
  this->transformed_data_.Reshape(top_shape);
  
  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.image_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  top_shape[0] = batch_size;
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
}

template <typename Dtype>
void FloDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void FloDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
  const int batch_size = image_data_param.batch_size();
  string root_folder = image_data_param.root_folder();

  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  int xSize, ySize;
  CHECK(readFloFile(root_folder + lines_[lines_id_].first, NULL, xSize, ySize))
      << "Could not load " << lines_[lines_id_].first;
  
  // Use data_transformer to infer the expected blob shape from a cv_img.
  vector<int> top_shape = vector<int>(4);
  top_shape[0] = 1;
  top_shape[1] = 2;
  top_shape[2] = ySize;
  top_shape[3] = xSize;
  
  //this->transformed_data_.Reshape(top_shape);
  
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();

  // datum scales
  const int lines_size = lines_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);
    
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image
    int offset = batch->data_.offset(item_id);
    //this->transformed_data_.set_cpu_data(prefetch_data + offset);
    
    CHECK(readFloFile(root_folder + lines_[lines_id_].first, prefetch_data + offset, xSize, ySize))
        << "Could not load " << lines_[lines_id_].first;
    
    //this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
    
    trans_time += timer.MicroSeconds();

    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.image_data_param().shuffle()) {
        ShuffleImages();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(FloDataLayer);
REGISTER_LAYER_CLASS(FloData);

}  // namespace caffe

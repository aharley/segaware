# Segmentation-Aware Convolutional Networks Using Local Attention Masks

[[Project Page]](http://www.cs.cmu.edu/~aharley/segaware/) [[Paper]](https://arxiv.org/abs/1708.04607)

<img src="http://www.cs.cmu.edu/~aharley/images/bike.png" width=493px>

### Installation
Our code follows the [DeepLabV2](https://bitbucket.org/aquariusjay/deeplab-public-ver2) setup almost exactly.

Follow the instructions for installing DeepLabV2, and then simply swap that Caffe with this Caffe.

### Basics
First, get a layer to produce embeddings. You can use RGB as your embedding, if you like.

Then, instead of doing standard convolution, with a prototxt like this:
```
layer {
  bottom: "conv1_1"
  top: "conv1_2"
  name: "conv1_2"
  type: "Convolution"
  param { lr_mult: 1 decay_mult: 1 } param { lr_mult: 2 decay_mult: 0 }
  convolution_param {
    num_output: 64
    pad: 1
    kernel_size: 3
  }
}
```

Do segmentation-aware convolution, with a prototxt like this:
```
layer {
  bottom: "conv1_1"
  top: "im2col_conv1_2"
  name: "im2col_conv1_2"
  type: "Im2col"
  convolution_param {
    num_output: 64 kernel_size: 3 pad: 1
    bottom_is_im2col: true
  }
}
layer { 
  bottom: "im2col_conv1_2" 
  bottom: "exp_emb_dist1_2_64chan"
  top: "im2col_conv1_2_hit"
  name: "im2col_conv1_2_hit"
  type: "Eltwise" 
  eltwise_param { operation: PROD }
}
layer { 
  bottom: "im2col_conv1_2_hit"
  top: "conv1_2"
  name: "conv1_2"
  type: "Convolution"
  param { lr_mult: ${MAINSCALE_LR1} decay_mult: 1 } param { lr_mult: ${MAINSCALE_LR2} decay_mult: 0 }
  convolution_param {
    num_output: 64 kernel_size: 3 pad: 1
    bottom_is_im2col: true
  }
}
```


### Citation
```
@inproceedings{harley_segaware,
  title = {Segmentation-Aware Convolutional Networks Using Local Attention Masks},
  author = {Adam W Harley, Konstantinos G. Derpanis, Iasonas Kokkinos},
  booktitle = {IEEE International Conference on Computer Vision (ICCV)},
  year = {2017},
}
```

### Help
Feel free to open issues on here! Also, I'm pretty good with email: aharley@cmu.edu

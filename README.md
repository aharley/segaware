# Segmentation-Aware Convolutional Networks Using Local Attention Masks

[[Project Page]](http://www.cs.cmu.edu/~aharley/segaware/) [[Paper]](https://arxiv.org/abs/1708.04607)

<img src="http://www.cs.cmu.edu/~aharley/images/bike.png" width=493px>

Segmentation-aware convolution filters are invariant to backgrounds. We achieve this in three steps: (i) compute segmentation cues for each pixel (i.e., “embeddings”), (ii) create a foreground mask for each patch, and (iii) combine the masks with convolution, so that the filters only process the local foreground in each image patch.

### Installation

For prerequisites, refer to [DeepLabV2](https://bitbucket.org/aquariusjay/deeplab-public-ver2). Our setup follows theirs almost exactly.

Once you have the prequisites, simply run `make all -j4` from within `caffe/` to compile the code with `4` cores.

### Learning embeddings with dedicated loss
- Use `Convolution` layers to create dense embeddings.
- Use `Im2dist` to compute dense distance comparisons in an embedding map.
- Use `Im2parity` to compute dense label comparisons in a label map.
- Use `DistLoss` (with parameters `alpha` and `beta`) to set up a contrastive side loss on the distances.

See `scripts/segaware/config/embs` for a full example.

### Setting up a segmentation-aware convolution layer
- Use `Im2col` on the input, to arrange pixel/feature patches into columns. 
- Use `Im2dist` on the embeddings, to get their distances into columns.
- Use `Exp` on the distances, with `scale: -1`, to get them into `[0,1]`.
- `Tile` the exponentiated distances, with a factor equal to the depth (i.e., channels) of the original convolution features.
- Use `Eltwise` to multiply the `Tile` result with the `Im2col` result.
- Use `Convolution` with `bottom_is_im2col: true` to matrix-multiply the convolution weights with the `Eltwise` output.

See `scripts/segaware/config/vgg` for an example in which every convolution layer in the VGG16 architecture is made segmentation-aware.

#### Using a segmentation-aware CRF
- Use the `NormConvMeanfield` layer. As input, give it two copies of the unary potentials (produced by a `Split` layer), some embeddings, and a meshgrid-like input (produced by a `DummyData` layer with `data_filler { type: "xy" }`).

See `scripts/segaware/config/res` for an example in which a segmentation-aware CRF is added to a resnet architecture.

### Replicating the segmentation results presented in our paper 

- Download pretrained model weights [here](https://drive.google.com/file/d/0B37FFJE7o45TbmVhT1AwVzR3bmM/view?usp=sharing), and put that file into `scripts/segaware/model/res/`.
- From `scripts`, run `./test_res.sh`. This will produce `.mat` files in `scripts/segaware/features/res/voc_test/mycrf/`.
- From `scripts`, run `./gen_preds.sh`. This will produce colorized `.png` results in `scripts/segaware/results/res/voc_test/mycrf/none/results/VOC2012/Segmentation/comp6_test_cls`. An example input-ouput pair is shown below:
<img src="http://www.cs.cmu.edu/~aharley/segaware/images/sample_results/2009_000776.jpg" width=369px>
<img src="http://www.cs.cmu.edu/~aharley/segaware/images/sample_results/2009_000776.png" width=369px>
- If you zip these results, and submit them to the official PASCAL VOC test server, you will get 79.83900% IOU.

If you run this set of steps for the validation set, you can run `./eval.sh` to evaluate your results on the PASCAL VOC validation set. If you change the model, you may want to run `./edit_env.sh` to update the evaluation instructions.

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

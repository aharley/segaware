% out_dir = '/home/aharley/caffe_scripts/segaware/results/fisher/voc_val/fc8/none/results/VOC2012/Segmentation/comp6_val_cls/front_python';
% out_dir2 = '/home/aharley/caffe_scripts/segaware/results/fisher/voc_val/fc8/none/results/VOC2012/Segmentation/comp6_val_cls/front_python2';
% out_dir = '/home/aharley/caffe_scripts/segaware/results/fisher/voc_val/fc8/none/results/VOC2012/Segmentation/comp6_val_cls/context_python';
% out_dir2 = '/home/aharley/caffe_scripts/segaware/results/fisher/voc_val/fc8/none/results/VOC2012/Segmentation/comp6_val_cls/context_python2';
out_dir = '/home/aharley/caffe_scripts/segaware/results/fisher/voc_val/fc8/none/results/VOC2012/Segmentation/comp6_val_cls/plusContext_python';
out_dir2 = '/home/aharley/caffe_scripts/segaware/results/fisher/voc_val/fc8/none/results/VOC2012/Segmentation/comp6_val_cls/plusContext_python2';

mkdir(out_dir2);

load('pascal_seg_colormap.mat');
imageList = dir(out_dir);
imageList = imageList(3:end);


for i=1:numel(imageList)
    fprintf('i=%d/%d\n',i,numel(imageList));
    im = imread(sprintf('%s/%s',out_dir,imageList(i).name));
%     imshow(im);
%     rgb = ind2rgb(im,colormap);
    x = rgb2ind(im,colormap);
    imwrite(x, colormap, sprintf('%s/%s',out_dir2,imageList(i).name));
%     pause;
end
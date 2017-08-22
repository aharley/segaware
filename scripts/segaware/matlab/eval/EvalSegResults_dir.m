SetupEnv;

VOC_root_folder = '/home/aharley/datasets/'
seg_root = fullfile(VOC_root_folder, 'VOC2012');
gt_dir   = fullfile(VOC_root_folder, 'VOC2012', 'SegmentationClass');

model_name = 'abcd';
%save_root_folder = fullfile('../../results/', model_name, testset);

save_root_folder = fullfile('/home/aharley/caffe_scripts/segaware/matlab/crfrnn_demo/abcd_out/mycrf/', model_name, testset);

trainset = 'train';
seg_res_dir = [save_root_folder '/results/VOC2012/'];

save_root_folder
seg_res_dir
% the pngs should be in here:
save_result_folder = fullfile(seg_res_dir, 'Segmentation', [id '_' testset '_cls']);

% get iou score
if strcmp(testset, 'voc_val')
  VOCopts = GetVOCopts(seg_root, seg_res_dir, trainset, 'val', 'VOC2012');
  [accuracies, avacc, conf, rawcounts] = MyVOCevalseg(VOCopts, id);
else
  fprintf(1, 'This is test set. No evaluation. Just saved as png\n');
end 

    
    



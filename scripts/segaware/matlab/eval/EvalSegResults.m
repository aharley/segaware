SetupEnv;

if strcmp(feature_type,'crf') 
  post_folder = sprintf('post_densecrf_W%d_XStd%d_RStd%d_PosW%d_PosXStd%d', bi_w, bi_x_std, bi_r_std, pos_w, pos_x_std);
else
  post_folder = 'none';
end

%map_folder = fullfile('/home/adam/final_code/deeplab-public_scripts_nofeats/feats', model_name, testset, feature_type, post_folder); 
%save_root_folder = fullfile('/home/adam/final_code/deeplab-public_scripts_nofeats/', dataset, 'res', model_name, testset, feature_type, post_folder);
save_root_folder = fullfile('../../results/', model_name, testset, feature_type, post_folder);

name = getComputerName();
if strcmp(name,'alpha') || strcmp(name,'beta') || strcmp(name,'gamma')
  VOC_root_folder = '/home/aharley/datasets/'
else
  VOC_root_folder = '/data2/datasets/'
end

%output_mat_folder = map_folder;

fprintf(1, 'Saving to %s\n', save_root_folder);

trainset = 'train';
seg_res_dir = [save_root_folder '/results/VOC2012/'];
seg_root = fullfile(VOC_root_folder, 'VOC2012');
gt_dir   = fullfile(VOC_root_folder, 'VOC2012', 'SegmentationClass');



% get iou score
if strcmp(testset, 'voc_val') || strcmp(testset, 'voc_reduced')
VOCopts = GetVOCopts(seg_root, seg_res_dir, trainset, testset(5:end), 'VOC2012');
save_result_folder = fullfile(seg_res_dir, 'Segmentation', [id '_' testset(5:end) '_cls']);
  [accuracies, avacc, conf, rawcounts] = MyVOCevalseg(VOCopts, id);
elseif strcmp(testset, 'voc_val_f')
VOCopts = GetVOCopts(seg_root, seg_res_dir, trainset, 'val', 'VOC2012');
save_result_folder = fullfile(seg_res_dir, 'Segmentation', [id '_' testset(5:end) '_cls']);
  [accuracies, avacc, conf, rawcounts] = MyVOCevalseg(VOCopts, id);
else
  fprintf(1, 'This is test set. No evaluation. Just saved as png\n');
end 

    
    



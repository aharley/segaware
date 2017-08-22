SetupEnv;

model_name = 'mycrf';
feature_type = 'crf';

deploy_path = '../crossval/deploys';
out_root = '../../results/mycrf/voc_val/crf'

deploys = dir([deploy_path '/*.prototxt']);
nDeploys = numel(deploys);
scores = zeros(nDeploys,1);

for dep=1:nDeploys
deploy = deploys(dep).name;
code = deploy(1:end-9);

save_root_folder = fullfile('../../results/', model_name, testset, feature_type, code);

VOC_root_folder = '/home/aharley/datasets/'

trainset = 'train';
seg_res_dir = [save_root_folder '/results/VOC2012/']
seg_root = fullfile(VOC_root_folder, 'VOC2012');
gt_dir   = fullfile(VOC_root_folder, 'VOC2012', 'SegmentationClass');

% get iou score
if strcmp(testset, 'voc_val')
  VOCopts = GetVOCopts(seg_root, seg_res_dir, trainset, 'val', 'VOC2012');
save_result_folder = fullfile(seg_res_dir, 'Segmentation', [id '_' 'val' '_cls']);
  [accuracies, avacc, conf, rawcounts] = MyVOCevalseg(VOCopts, id);
else
  fprintf(1, 'This is test set. No evaluation. Just saved as png\n');
end
scores(dep) = avacc;

clc;
fprintf('top 10 scores for deploys %d/%d\n',dep,nDeploys);
[sortedScores, inds] = sort(scores,'descend');
for deep=1:60
fprintf('%s: %.3f\n',deploys(inds(deep)).name,sortedScores(deep));
end

end

save('crossval_partial_results2.mat','scores','deploys');

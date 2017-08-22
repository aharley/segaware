%addpath('/home/adam/final_code/deeplab-public/matlab/my_script');
SetupEnv;

if strcmp(feature_type,'crf') 

disp('ok! generating preds from crfs');

post_folder = sprintf('post_densecrf_W%d_XStd%d_RStd%d_PosW%d_PosXStd%d', bi_w, bi_x_std, bi_r_std, pos_w, pos_x_std);
map_folder = fullfile('/home/aharley/caffe_scripts/segaware/features/', model_name, testset, feature_type, post_folder); 
save_root_folder = fullfile('/home/aharley/caffe_scripts/segaware/results/', model_name, testset, feature_type, post_folder);

map_dir = dir(fullfile(map_folder, '*.bin'));

fprintf(1,' saving to %s\n', save_root_folder);

seg_res_dir = [save_root_folder '/results/VOC2012/'];

%save_result_folder = fullfile(seg_res_dir, 'Segmentation', [id '_' testset '_cls']);
save_result_folder = fullfile(seg_res_dir, 'Segmentation', [id '_' testset(5:end) '_cls']);

if ~exist(save_result_folder, 'dir')
    mkdir(save_result_folder);
end

for i = 1 : numel(map_dir)
    fprintf(1, 'generating preds from crf for image %d (%d)...\n', i, numel(map_dir));
    map = LoadBinFile(fullfile(map_folder, map_dir(i).name), 'int16');

    img_fn = map_dir(i).name(1:end-4);
    imwrite(uint8(map), colormap, fullfile(save_result_folder, [img_fn, '.png']));
end

else %if (strcmp(feature_type,'fc8') || strcmp(feature_type,'fc8_safe'))


disp('ok! generating preds');

mat_folder  = fullfile('../../features/', model_name, testset, feature_type)
name = getComputerName();
if strcmp(name,'alpha') || strcmp(name,'beta') || strcmp(name,'gamma')
  img_folder  = '/home/aharley/datasets/VOC2012/JPEGImages';
else
  img_folder  = '/data2/datasets/VOC2012/JPEGImages';
end

post_folder = 'none';
save_root_folder = fullfile('../../results/', model_name, testset, feature_type, post_folder);

fprintf(1,' saving to %s\n', save_root_folder);

seg_res_dir = [save_root_folder '/results/VOC2012/'];

%if (strcmp(testset,'voc_val'))
%save_result_folder = fullfile(seg_res_dir, 'Segmentation', [id '_' 'val' '_cls']);
%elseif (strcmp(testset,'voc_test'))
%save_result_folder = fullfile(seg_res_dir, 'Segmentation', [id '_' 'test' '_cls']);
%else
%save_result_folder = fullfile(seg_res_dir, 'Segmentation', [id '_' testset '_cls']);
%end
save_result_folder = fullfile(seg_res_dir, 'Segmentation', [id '_' testset(5:end) '_cls']);

if ~exist(save_result_folder, 'dir')
    mkdir(save_result_folder);
end

mat_dir = dir(fullfile(mat_folder, '*.mat'));
fprintf(1, 'found %d mats in %s\n', numel(mat_dir), mat_folder);
%parpool('local',8);
%parfor i = 1 : numel(mat_dir)
for i = 1 : numel(mat_dir)
    fprintf(1, 'generating preds for image %d (%d)...\n', i, numel(mat_dir));

    img_fn = mat_dir(i).name(1:end-4);
    img_fn = strrep(img_fn, '_blob_0', '');
    img_fn = strrep(img_fn, '_blob_1', '');
if overwriteResults || (exist(fullfile(save_result_folder, [img_fn, '.png']), 'file') ~= 2)
    data = load(fullfile(mat_folder, mat_dir(i).name));
    data = data.data;
    data = permute(data, [2 1 3]);
    % Transform data to probability
    data = exp(data);
    data = bsxfun(@rdivide, data, sum(data, 3));

    img = imread(fullfile(img_folder, [img_fn, '.jpg']));
    img_row = size(img, 1);
    img_col = size(img, 2);
    
    data = data(1:img_row, 1:img_col, :);

    [~,classes] = max(data,[],3);
    classes = classes-1;

    imwrite(uint8(classes), colormap, fullfile(save_result_folder, [img_fn, '.png']));
end
end

end

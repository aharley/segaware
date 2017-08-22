SetupEnv;
testset='voc_val';
%model_name = 'mycrf';
%feature_type = 'crf';

model_name = 'my-resnet-101';
feature_type = 'fc1';

%result_dir = '../../results/mycrf/voc_val/crf';
result_dir = '../../results/my-resnet-101/voc_val/fc1';
result_dirs = dir([result_dir '/000*']);
nResults = numel(result_dirs);

scores = zeros(nResults,1);
codes = zeros(nResults,1);
for res=1:nResults
    code = result_dirs(res).name;
    codes(res) = str2double(code);
    matPath = ['scores/' code '.mat'];
%if (findstr(code,'k15_d09'))
%if (findstr(code,'sxy0000_srgb1300'))
%if (findstr(code,'sxy0000_k05'))
    if exist(matPath, 'file') == 2
        fprintf('skipping %s\n', code);
        load(matPath);
        scores(res) = score;
    else
        save_root_folder = fullfile('../../results/', model_name, testset, feature_type, code);
        
        VOC_root_folder = '../../../../datasets/';
        
        trainset = 'train';
        seg_res_dir = [save_root_folder '/results/VOC2012/'];
        fprintf('hey, %s is new!\n', code);
        seg_root = fullfile(VOC_root_folder, 'VOC2012');
        gt_dir   = fullfile(VOC_root_folder, 'VOC2012', 'SegmentationClass');
        
        % get iou score
        VOCopts = GetVOCopts(seg_root, seg_res_dir, trainset, 'val', 'VOC2012');
        save_result_folder = fullfile(seg_res_dir, 'Segmentation', [id '_' 'val' '_cls']);
        
        nPngs = numel(dir([save_result_folder '/*.png']));
        if nPngs==346
            [accuracies, avacc, conf, rawcounts] = MyVOCevalseg(VOCopts, id);
            score = avacc;
            scores(res) = score;
            save(matPath,'score','code');
        else
            fprintf('%s is not complete\n', code);
        end
%	pause;
    clc;
    fprintf('top scores for models %d/%d\n',res,nResults);
    %[sortedScores, inds] = sort(scores,'descend');
    [sortedCodes, inds] = sort(codes,'ascend');
    for r=1:min(numel(codes),50)
        fprintf('%02d) %s: %.3f\n',r,result_dirs(inds(r)).name,sortedCodes(r));
    end

    end
end
%end
%end
%end
    clc;
    fprintf('top scores for models %d/%d\n',res,nResults);
    %[sortedScores, inds] = sort(scores,'descend');
    [sortedCodes, inds] = sort(codes,'ascend');
    for r=1:min(numel(codes),50)
	    fprintf('%02d) %06d: %.3f\n',r,sortedCodes(r),scores(inds(r)));
    end

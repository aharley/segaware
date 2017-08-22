model_name = 'res';

%feature_type = 'fc1'; 
%feature_type = 'fc8'; 
feature_type = 'mycrf'; 

%testset    = 'voc_val';
testset    = 'voc_test';

overwriteResults = 0;

% set up the environment variables
load('./pascal_seg_colormap.mat');
is_server       = 0;
crf_load_mat    = 0;   % the densecrf code load MAT files directly (no call SaveMatAsBin.m)
                       % used ONLY by DownSampleFeature.m
learn_crf       = 0;   % NOT USED. Set to 0
is_mat          = 1;   % the results to be evaluated are saved as mat (1) or png (0)
has_postprocess = 0;   % has done densecrf post processing (1) or not (0)
is_argmax       = 0;   % the output has been taken argmax already (e.g., coco dataset). 
                       % assume the argmax takes C-convention (i.e., start from 0)
debug           = 0;   % if debug, show some results
% deep msc coco largeFOV values for crf
bi_w           = 3; 
bi_x_std       = 77;
bi_y_std       = 77;
bi_r_std       = 4;
bi_g_std       = 4;
bi_b_std       = 4;
pos_w          = 2;
pos_x_std      = 2;
pos_y_std      = 2;
dataset    = 'voc12'; 
id           = 'comp6';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% used for cross-validation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(10)

% downsampling files for cross-validation
down_sample_method = 2;      % 1: equally sample with "down_sample_rate", 2: randomly pick "num_sample" samples
down_sample_rate   = 8;
num_sample         = 100;    % number of samples used for cross-validation

% ranges for cross-validation
range_pos_w = [3];
range_pos_x_std = [3];

range_bi_w = [5];
range_bi_x_std = [49];
range_bi_r_std = [4 5];



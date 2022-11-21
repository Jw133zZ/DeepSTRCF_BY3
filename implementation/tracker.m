function results = tracker(params)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
global enableGPU;
enableGPU = true;
% Get sequence info
[seq, im] = get_sequence_info(params.seq);
params = rmfield(params, 'seq');
if isempty(im)
    seq.rect_position = [];
    [~, results] = get_sequence_results(seq);
    return;
end
resp_w = params.resp_w;
% Init position
pos = seq.init_pos(:)';
target_sz = seq.init_sz(:)';
params.init_sz = target_sz;
layers = params.layers;
% Channel Weight
mean_factor = params.mean_factor;
mean_yz = params.mean_yz;

% Feature settings
features = params.t_features;

% Set default parameters
% params = init_default_params(params);

% Global feature parameters
if isfield(params, 't_global')
    global_fparams = params.t_global;
else
    global_fparams = [];
end

global_fparams.use_gpu = params.use_gpu;
global_fparams.gpu_id = params.gpu_id;

% Define data types
if params.use_gpu
    params.data_type = zeros(1, 'double', 'gpuArray');
else
    params.data_type = zeros(1, 'double');
end
params.data_type_complex = complex(params.data_type);

global_fparams.data_type = params.data_type;

% Load learning parameters
admm_max_iterations = params.max_iterations;
init_penalty_factor = params.init_penalty_factor;
max_penalty_factor = params.max_penalty_factor;
penalty_scale_step = params.penalty_scale_step;
temporal_regularization_factor = params.temporal_regularization_factor;

init_target_sz = target_sz;
% Check if color image
if size(im,3) == 3
    if all(all(im(:,:,1) == im(:,:,2)))
        is_color_image = false;
    else
        is_color_image = true;
    end
else
    is_color_image = false;
end

if size(im,3) > 1 && is_color_image == false
    im = im(:,:,1);
end

% Check if mexResize is available and show warning otherwise.
params.use_mexResize = true;
global_fparams.use_mexResize = true;
try
    [~] = mexResize(ones(5,5,3,'uint8'), [3 3], 'auto');
catch err
    params.use_mexResize = false;
    global_fparams.use_mexResize = false;
end

% Calculate search area and initial scale factor
search_area = prod(init_target_sz * params.search_area_scale);
if search_area > params.max_image_sample_size
    currentScaleFactor = sqrt(search_area / params.max_image_sample_size);
elseif search_area < params.min_image_sample_size
    currentScaleFactor = sqrt(search_area / params.min_image_sample_size);
else
    currentScaleFactor = 1.0;
end

% target size at the initial scale
base_target_sz = target_sz / currentScaleFactor;

% window size, taking padding into account
switch params.search_area_shape
    case 'proportional'
        img_sample_sz = floor(base_target_sz * params.search_area_scale);     % proportional area, same aspect ratio as the target
    case 'square'
        img_sample_sz = repmat(sqrt(prod(base_target_sz * params.search_area_scale)), 1, 2); % square area, ignores the target aspect ratio
    case 'fix_padding'
        img_sample_sz = base_target_sz + sqrt(prod(base_target_sz * params.search_area_scale) + (base_target_sz(1) - base_target_sz(2))/4) - sum(base_target_sz)/2; % const padding
    case 'custom'
        img_sample_sz = [base_target_sz(1)*2 base_target_sz(2)*2];
end

[features, global_fparams, feature_info] = init_features(features, global_fparams, is_color_image, img_sample_sz, 'exact');

% Set feature info
img_support_sz = feature_info.img_support_sz;
feature_sz = unique(feature_info.data_sz, 'rows', 'stable');
feature_cell_sz = unique(feature_info.min_cell_size, 'rows', 'stable');
num_feature_blocks = size(feature_sz, 1);

% Get feature specific parameters
feature_extract_info = get_feature_extract_info(features);

% Size of the extracted feature maps
feature_sz_cell = mat2cell(feature_sz, ones(1,num_feature_blocks), 2);
filter_sz = feature_sz;
filter_sz_cell = permute(mat2cell(filter_sz, ones(1,num_feature_blocks), 2), [2 3 1]);
down_use_sz = filter_sz_cell{2};
% The size of the label function DFT. Equal to the maximum filter size
[output_sz, k1] = max(filter_sz, [], 1);
k1 = k1(1);

% Get the remaining block indices
block_inds = 1:num_feature_blocks;
block_inds(k1) = [];

% Construct the Gaussian label function
yf = cell(numel(num_feature_blocks), 1);
for i = 1:num_feature_blocks
    sz = filter_sz_cell{i};
    output_sigma = sqrt(prod(floor(base_target_sz/feature_cell_sz(i)))) * params.output_sigma_factor;
    rg           = circshift(-floor((sz(1)-1)/2):ceil((sz(1)-1)/2), [0 -floor((sz(1)-1)/2)]);
    cg           = circshift(-floor((sz(2)-1)/2):ceil((sz(2)-1)/2), [0 -floor((sz(2)-1)/2)]);
    [rs, cs]     = ndgrid(rg,cg);
    y            = exp(-0.5 * (((rs.^2 + cs.^2) / output_sigma^2)));
    yf{i}           = fft2(y);
end
yf = {yf{1},yf{2},yf{2}};
% Compute the cosine windows
cos_window = cellfun(@(sz) hann(sz(1))*hann(sz(2))', feature_sz_cell, 'uniformoutput', false);
cos_window = {cos_window{1};cos_window{2};cos_window{2}};

% Define spatial regularization windows
reg_window = cell(num_feature_blocks, 1);
for i = 1:num_feature_blocks
    reg_scale = floor(base_target_sz/params.feature_downsample_ratio(i));
    use_sz = filter_sz_cell{i};
    reg_window{i} = ones(use_sz) * params.reg_window_max;
    range = zeros(numel(reg_scale), 2);
    
    % determine the target center and range in the regularization windows
    for j = 1:numel(reg_scale)
        range(j,:) = [0, reg_scale(j) - 1] - floor(reg_scale(j) / 2);
    end
    center = floor((use_sz + 1)/ 2) + mod(use_sz + 1,2);
    range_h = (center(1)+ range(1,1)) : (center(1) + range(1,2));
    range_w = (center(2)+ range(2,1)) : (center(2) + range(2,2));
    
    reg_window{i}(range_h, range_w) = params.reg_window_min;
end
reg_window = {reg_window{1};reg_window{2};reg_window{2}};
% Pre-computes the grid that is used for socre optimization
ky = circshift(-floor((filter_sz_cell{1}(1) - 1)/2) : ceil((filter_sz_cell{1}(1) - 1)/2), [1, -floor((filter_sz_cell{1}(1) - 1)/2)]);
kx = circshift(-floor((filter_sz_cell{1}(2) - 1)/2) : ceil((filter_sz_cell{1}(2) - 1)/2), [1, -floor((filter_sz_cell{1}(2) - 1)/2)])';
newton_iterations = params.newton_iterations;

% Use the translation filter to estimate the scale
nScales = params.number_of_scales;
scale_step = params.scale_step;
scale_exp = (-floor((nScales-1)/2):ceil((nScales-1)/2));
scaleFactors = scale_step .^ scale_exp;
if nScales > 0
    %force reasonable scale changes
    min_scale_factor = scale_step ^ ceil(log(max(5 ./ img_support_sz)) / log(scale_step));
    max_scale_factor = scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ base_target_sz)) / log(scale_step));
end

seq.time = 0;

% Define the learning variables
f_pre_f = cell(num_feature_blocks+1, 1);
cf_f = cell(num_feature_blocks+1, 1);
x_vgg16 = cell(num_feature_blocks,1);
x_deep = cell(num_feature_blocks,1);
W_channels = cell(num_feature_blocks,1);
index = cell(num_feature_blocks,1);
% W_response = cell(1,1,num_feature_blocks+1);
% for i=1:3
%     W_response{i}=resp_w(i);
% end
% Allocate
scores_fs_feat = cell(1,1,num_feature_blocks+1);
while true
    % Read image
    if seq.frame > 0
        [seq, im] = get_sequence_frame(seq);
        if isempty(im)
            break;
        end
        if size(im,3) > 1 && is_color_image == false
            im = im(:,:,1);
        end
    else
        seq.frame = 1;
    end
    
    tic();
    
%     if seq.frame ==28
%         dlmwrite('E:\tracker_benchmark_v1.0\trackers\DeepSTRCF_BY3\x.txt', true);
%     end
%     
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Target localization step
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Do not estimate translation and scaling on the first frame, since we
    % just want to initialize the tracker there
    if seq.frame > 1
        old_pos = inf(size(pos));
        iter = 1;
        by_iter = 1;
        %translation search
        while iter <= params.refinement_iterations && any(old_pos ~= pos)
            while true
                % calculating the position of HOG_CN
                sample_pos = round(pos);
                sample_scale = currentScaleFactor*scaleFactors;
                xt = extract_features(im, sample_pos, sample_scale ,...
                    {features{1},features{2},features{3}}, global_fparams, feature_extract_info);
                [deep_pixels, ~, ~] = get_pixels(im,sample_pos,round(img_sample_sz*sample_scale(3)),img_sample_sz);
                for i=1:2
                    x_vgg16{i} = normalization(get_vggfeatures(deep_pixels,down_use_sz,layers(i)));
                    x_vgg16{i} = x_vgg16{i}(:,:,index{i});
                    x_vgg16{i} = bsxfun(@times,x_vgg16{i},W_channels{i});
                end
                xt = {xt{1};x_vgg16{1};x_vgg16{2}};
                %xt{1}:1+31+10    x_vgg16{1}:deep   x_vgg16{2}:middle
                xtw = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xt, cos_window, 'uniformoutput', false);
                xtf = cellfun(@fft2, xtw, 'uniformoutput', false);
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %博弈过程
                if by_iter==1
                    %position of hog+cn
                    scores_hc = gather(sum(bsxfun(@times, conj(cf_f{1}), xtf{1}), 3));
                    scores_deep = resizeDFT2(gather(sum(bsxfun(@times, conj(cf_f{2}), xtf{2}), 3)),output_sz);
                    scores_middle = resizeDFT2(gather(sum(bsxfun(@times, conj(cf_f{3}), xtf{3}), 3)),output_sz);
                    scores_hd = scores_hc.*resp_w(1) + scores_deep.*resp_w(2);
                    scores_hm = scores_hc.*resp_w(1) + scores_middle.*resp_w(3);
                    
                    %%%%%分别进行博弈
                    scores_fs_hd = permute(gather(scores_hd),[1 2 4 3]);
                    responsef_padded_hd = resizeDFT2(scores_fs_hd, output_sz);
                    response_hd = ifft2(responsef_padded_hd, 'symmetric');
                    [disp_row_hd, disp_col_hd, sind] = resp_newton(response_hd, responsef_padded_hd, newton_iterations, ky, kx, output_sz);
                    scale_change_factor = scaleFactors(sind);
                    translation_vec_hd = [disp_row_hd, disp_col_hd] .* (img_support_sz./output_sz) * currentScaleFactor * scaleFactors(sind);
                    pos_hd = sample_pos + translation_vec_hd;
                    %limit
                    pos_hd = max([1 1], min([size(im,1) size(im,2)], pos_hd));
                    
                    %position of vgg_deep
                    scores_fs_hm = permute(gather(scores_hm),[1 2 4 3]);
                    responsef_padded_hm = resizeDFT2(scores_fs_hm, output_sz);
                    response_hm = ifft2(responsef_padded_hm, 'symmetric');
                    [disp_row_hm, disp_col_hm, ~] = resp_newton(response_hm, responsef_padded_hm, newton_iterations, ky, kx, output_sz);
                    translation_vec_hm = [disp_row_hm, disp_col_hm] .* (img_support_sz./output_sz) * currentScaleFactor * scaleFactors(sind);
                    pos_hm = sample_pos + translation_vec_hm;
                    %limit
                    pos_hm = max([1 1], min([size(im,1) size(im,2)], pos_hm));
                    if sqrt((pos_hm(1)-pos_hd(1)).^2+(pos_hm(2)-pos_hd(2)).^2)<params.pos_diff
                        pos = (pos_hm+pos_hd)/2;
                        pos = max([1 1], min([size(im,1) size(im,2)], pos));                        
                        old_pos = pos;
                        break;
                    end
                else
                    scores_hc = gather(sum(bsxfun(@times, conj(cf_f{1}), xtf{1}), 3));
                    scores_deep = resizeDFT2(gather(sum(bsxfun(@times, conj(cf_f{2}), xtf{2}), 3)),output_sz);
                    scores_middle = resizeDFT2(gather(sum(bsxfun(@times, conj(cf_f{3}), xtf{3}), 3)),output_sz);
                    scores_hd = scores_hc.*resp_w(1) + scores_deep.*resp_w(2);
                    scores_hm = scores_hc.*resp_w(1) + scores_middle.*resp_w(3);    
                    %计算权重
                    appr_hd =  cal_psr(scores_hd,scores_hc);
                    appr_hm =  cal_psr(scores_hm,scores_hc);
                    w_hd = appr_hd / (appr_hd + appr_hm);
                    w_hm = appr_hm / (appr_hd + appr_hm);
                    %%%%各自加权
                    scores_hd = scores_hd + scores_hm.*w_hm ;
                    scores_hm = scores_hm + scores_hd.*w_hd ;
                    
                    scores_fs_hm = permute(gather(scores_hm), [1 2 4 3]);
                    scores_fs_hd = permute(gather(scores_hd), [1 2 4 3]);
                    
                    responsef_padded_hd = resizeDFT2(scores_fs_hd, output_sz);
                    responsef_padded_hm = resizeDFT2(scores_fs_hm, output_sz);
  
                    response_hd = ifft2(responsef_padded_hd, 'symmetric');
                    response_hm = ifft2(responsef_padded_hm, 'symmetric');
            
                    
                    %position of hd
                    [disp_row_hd, disp_col_hd, sind] = resp_newton(response_hd, responsef_padded_hd, newton_iterations, ky, kx, output_sz);
                    scale_change_factor = scaleFactors(sind);
                    translation_vec_hd = [disp_row_hd, disp_col_hd] .* (img_support_sz./output_sz) * currentScaleFactor * scaleFactors(sind);
                    pos_hd = sample_pos + translation_vec_hd;
                    pos_hd = max([1 1], min([size(im,1) size(im,2)], pos_hd));
                    %position of hm
                    [disp_row_hm, disp_col_hm, ~] = resp_newton(response_hm, responsef_padded_hm, newton_iterations, ky, kx, output_sz);
                    translation_vec_hm = [disp_row_hm, disp_col_hm] .* (img_support_sz./output_sz) * currentScaleFactor * scaleFactors(sind);
                    pos_hm = sample_pos + translation_vec_hm;
                    pos_hm = max([1 1], min([size(im,1) size(im,2)], pos_hm));
                    
                    pos = (pos_hd + pos_hm)/2;
                    pos = max([1 1], min([size(im,1) size(im,2)], pos));
                    
                    if by_iter > params.maxby_iters || sqrt((pos_hm(1)-pos_hd(1)).^2+(pos_hm(2)-pos_hd(2)).^2)<params.pos_diff
                        old_pos = pos;
                        break;
                    end
                end
                by_iter = by_iter+1;
            end
            % Update the scale
            currentScaleFactor = currentScaleFactor * scale_change_factor;
            
            % Adjust to make sure we are not to large or to small
            if currentScaleFactor < min_scale_factor
                currentScaleFactor = min_scale_factor;
            elseif currentScaleFactor > max_scale_factor
                currentScaleFactor = max_scale_factor;
            end
            
            iter = iter + 1;
            
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Model update step
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % extract image region for training sample
    sample_pos = round(pos);
    xl = extract_features(im, sample_pos, currentScaleFactor, ...
        {features{1},features{2},features{3}}, global_fparams, feature_extract_info);
    [pixels, pos_bg, resize_factor] = get_pixels(im,pos,round(img_sample_sz*currentScaleFactor),img_sample_sz);
    for i=1:2
        x_vgg16{i} = get_vggfeatures(pixels,down_use_sz,layers(i));
        x_vgg16{i} = normalization(x_vgg16{i});
    end
    if seq.frame==1
        for i=1:2
            x_deep{i} = normalization(get_vggfeatures(pixels,output_sz,layers(i)));
            x_deep{i} = fft2(bsxfun(@times,x_deep{i},cos_window{1}));
            xw_vgg16 = real(ifft2(x_deep{i}));
            [W_channels{i},index{i}] = testBFAER(xw_vgg16, [size(pixels, 1) size(pixels, 2)], ...
                pos, target_sz, pos_bg, resize_factor,mean_factor(i),mean_yz(i));
        end
    end
    for i=1:2
        x_vgg16{i} = x_vgg16{i}(:,:,index{i}) ;
        x_vgg16{i} = bsxfun(@times,x_vgg16{i},W_channels{i});
    end
    xl = {xl{1};x_vgg16{1};x_vgg16{2}};
    % do windowing of features
    xlw = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xl, cos_window, 'uniformoutput', false);
    
    % compute the fourier series
    xlf = cellfun(@fft2, xlw, 'uniformoutput', false);
    
    % train the CF model for each feature
    for k = 1: numel(xlf)
        model_xf = xlf{k};
        
        if (seq.frame == 1)
            f_pre_f{k} = zeros(size(model_xf));
            mu = 0;
        else
            mu = temporal_regularization_factor(k);
        end
        
        % intialize the variables
        f_f = double(zeros(size(model_xf)));
        g_f = f_f;
        h_f = f_f;
        gamma  = init_penalty_factor(k);
        gamma_max = max_penalty_factor(k);
        gamma_scale_step = penalty_scale_step(k);
        
        % use the GPU mode
        if params.use_gpu
            model_xf = gpuArray(model_xf);
            f_f = gpuArray(f_f);
            f_pre_f{k} = gpuArray(f_pre_f{k});
            g_f = gpuArray(g_f);
            h_f = gpuArray(h_f);
            reg_window{k} = gpuArray(reg_window{k});
            yf{k} = gpuArray(yf{k});
        end
        
        % pre-compute the variables
        T = prod(output_sz);
        S_xx = sum(conj(model_xf) .* model_xf, 3);
        Sf_pre_f = sum(conj(model_xf) .* f_pre_f{k}, 3);
        Sfx_pre_f = bsxfun(@times, model_xf, Sf_pre_f);
        
        % solve via ADMM algorithm
        iter = 1;
        while (iter <= admm_max_iterations)
            
            % subproblem f
            B = S_xx + T * (gamma + mu);
            Sgx_f = sum(conj(model_xf) .* g_f, 3);
            Shx_f = sum(conj(model_xf) .* h_f, 3);
            
            f_f = ((1/(T*(gamma + mu)) * bsxfun(@times,  yf{k}, model_xf)) - ((1/(gamma + mu)) * h_f) +(gamma/(gamma + mu)) * g_f) + (mu/(gamma + mu)) * f_pre_f{k} - ...
                bsxfun(@rdivide,(1/(T*(gamma + mu)) * bsxfun(@times, model_xf, (S_xx .*  yf{k})) + (mu/(gamma + mu)) * Sfx_pre_f - ...
                (1/(gamma + mu))* (bsxfun(@times, model_xf, Shx_f)) +(gamma/(gamma + mu))* (bsxfun(@times, model_xf, Sgx_f))), B);
            
            %   subproblem g
            g_f = fft2(argmin_g(reg_window{k}, gamma, real(ifft2(gamma * f_f+ h_f)), g_f));
            
            %   update h
            h_f = h_f + (gamma * (f_f - g_f));
            
            %   update gamma
            gamma = min(gamma_scale_step * gamma, gamma_max);
            
            iter = iter+1;
        end
        
        % save the trained filters
        f_pre_f{k} = f_f;
        cf_f{k} = f_f;
    end
    
    % Update the target size (only used for computing output box)
    target_sz = base_target_sz * currentScaleFactor;

    %save position and calculate FPS
    tracking_result.center_pos = double(pos);
    tracking_result.target_size = double(target_sz);
    seq = report_tracking_result(seq, tracking_result);
    
    seq.time = seq.time + toc();
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Visualization
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if seq.frame == 9
    end
    
    
    % visualization
    if params.visualization
        rect_position_vis = [pos([2,1]) - (target_sz([2,1]) - 1)/2, target_sz([2,1])];
        im_to_show = double(im)/255;
        if size(im_to_show,3) == 1
            im_to_show = repmat(im_to_show, [1 1 3]);
        end
        if seq.frame==1
            fig_handle = figure('Name', 'Tracking');
            imagesc(im_to_show);
            hold on;
            rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
            text(10, 10, [int2str(seq.frame) '/'  int2str(size(seq.image_files, 1))], 'color', [0 1 1]);
            hold off;
            axis off;axis image;set(gca, 'Units', 'normalized', 'Position', [0 0 1 1])
        else
            figure(fig_handle);
            imagesc(im_to_show);
            hold on;
            rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
            text(10, 10, [int2str(seq.frame) '/'  int2str(size(seq.image_files, 1))], 'color', [0 1 1]);
            hold off;
            axis off;axis image;set(gca, 'Units', 'normalized', 'Position', [0 0 1 1])
        end
        drawnow
    end
end

[~, results] = get_sequence_results(seq);

disp(['fps: ' num2str(results.fps)])


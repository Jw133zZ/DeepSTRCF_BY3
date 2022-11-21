function [W_channel, index] = testBFAER(feat, patch_sz, pos, target_sz, pos_bg, resize_factor,mean_factor,...
                                         mean_yz)

%% set parameters and initialization
% num_efficient = zeros(1, size(feat, 3));
% count = 0;             %有效通道的数量
BFAER = zeros(1, size(feat, 3));

%% Resize feature size to patch size
feat = imResample(gather(feat), patch_sz);

%% locate the target area and search area
rect_tgt = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
% rect_bg = [pos([2,1]) - bg_area([2,1])/2, bg_area([2,1])];
rect_bg = [pos_bg(1), pos_bg(2), patch_sz];
%pos_bg

rect_tgt_resized = round([rect_tgt(1)-rect_bg(1), rect_tgt(2)-rect_bg(2), target_sz([2, 1])] * resize_factor);

xs = rect_tgt_resized(1):rect_tgt_resized(1)+rect_tgt_resized(3);
ys = rect_tgt_resized(2):rect_tgt_resized(2)+rect_tgt_resized(4);

xs(xs < 1) = 1;
ys(ys < 1) = 1;
xs(xs > size(feat,2)) = size(feat,2);
ys(ys > size(feat,1)) = size(feat,1);

%% calculate the background-foreground average energy ratio(BFAER)
for i = 1:size(feat, 3) % for each channel
    
    feat_bg = feat(:, :, i);
    feat_bg(ys, xs) = 0;
    
    AE_background = mean(mean(feat_bg(:, :)));
    AE_foreground = mean(mean(feat(ys, xs, i)));
    
    BFAER(i) = AE_foreground / AE_background;
    
%     if BFAER(i) >= 5
%         count = count + 1;
%         
% %         figure, imagesc(feat(:,:,i));
%     end    
    % visualization
%     figure, imagesc(feat(:,:,i));
%     rect_handle = rectangle('Position',rect_tgt_resized, 'EdgeColor','r','LineWidth',3);hold on;
%     text_handle = text(5, 10, ['channel=' int2str(i) '      BFAER=' num2str(BFAER(i))]);
%     set(text_handle, 'color', [1 1 0], 'FontSize', 15);
%     f = getframe(gca);
%     imwrite(f.cdata, ['E:\temp\BFAER_feature_maps\fig_' num2str(i) '.png']);
%     close;
end
% disp(num2str(num_efficient));

 %% sort BFAER by descend
  W_channel=(BFAER./(mean(BFAER)*mean_factor));
  [~,index] = find(W_channel>mean_yz);
  W_channel = W_channel(1,index);
  
  
  W_channel=reshape(W_channel,[1,1,numel(index)]);
% [BFAER, index] = sort(BFAER, 'descend');
% BFAER = reshape(BFAER,[],8); 
%% select top C channels for tracking
% selected_channels = index(1:num_channels);

end
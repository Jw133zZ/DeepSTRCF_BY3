function feat=normalization(x)
%          feat=bsxfun(@times,x,(size(x,1)*size(x,2)*size(x,3)./...
%             (sum(abs(reshape(x, [], 1, 1, size(x,4))).^2, 1) + eps)).^(1/2));
        feat=(x-min(x(:)))/(max(x(:))-min(x(:)));
end
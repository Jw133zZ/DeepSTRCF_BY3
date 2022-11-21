% PSR = (Fmax - mean(response)) / std(response)
function psr = cal_psr(response,resp_hc)

response_temp = real(response);
% response_temp = response;
real_hc = real(resp_hc);
Fmax = max(response_temp(:));
% Fmax = sqrt(abs(Fmax).^2);
% Fmin = min(response_temp(:));

%  psr = ((Fmax-Fmin).^2)/((mean(sum((response_temp(:,:)-Fmin).^2))));

%  psr = (Fmax - mean2(real_hc)) / std2(response_temp);
 psr = (Fmax - mean2(real_hc))   /  mean2(real_hc) + mean2(response_temp);
end
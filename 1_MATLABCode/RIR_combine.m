idx = 0:5:495;
RIR = zeros(32,8000,length(idx));
% Ys = zeros(16, length(idx));
for ii = 1:length(idx)
    file_name = ['RIR_Data_0_order/' 'Anm_FIR_0order_Rand_' num2str(idx(ii)) 'azi_']
    vaRR = load(file_name);
    RIR(:,:,ii) = vaRR.h1(:,1:8000);
%     Ys(:,ii) = vaRR.Ys;
end

% save('RIR_Ys.mat','RIR', 'Ys');
save('RIR_0_order.mat','RIR');
azi_ax = [0:5:495];
RIR = zeros(32,16000,length(azi_ax));
for ii = 1:length(azi_ax)
    file_name = ['Anm_FIR_6order_Rand_' num2str(azi_ax(ii)) 'azi_']
    vaRR = load(file_name);
    RIR(:,:,ii) = vaRR.h1;
end

save('RIR.mat','RIR');
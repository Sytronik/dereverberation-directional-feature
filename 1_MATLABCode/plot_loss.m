% final_epoch = 60
% clear;
% loss_struct{1} = load('MLP_pReLU_loss_39.mat');
% loss_struct{1} = load('MLP_loss_59.mat');
sum_I_a = false;
only_all = true;
loss_struct{1}.loss_train = loss_train;
loss_struct{1}.loss_valid = loss_valid;
loss_struct{1}.snr_seg_valid = snr_seg_valid;


epochs = 1:50;


for ii = 1:numel(loss_struct)
    [~, N] = size(loss_struct{ii}.loss_valid);
    if N == 2 || N == 3
        if only_all
            loss_struct{ii}.loss_valid = loss_struct{ii}.loss_valid(:,3).';
            loss_struct{ii}.snr_seg_valid = loss_struct{ii}.snr_seg_valid(:,3).';

        else
            loss_struct{ii}.loss_valid = loss_struct{ii}.loss_valid.';
            loss_struct{ii}.snr_seg_valid = loss_struct{ii}.snr_seg_valid.';
        end
    end
    [M, ~] = size(loss_struct{ii}.loss_valid);
    if sum_I_a && M ==2
        loss_valid(1, :) = sum(loss_struct{ii}.loss_valid(1:2, :), 1);
        loss_valid(2:3, :) = loss_struct{ii}.loss_valid(1:2, :);
        loss_struct{ii}.loss_valid = loss_valid;
    end
%     loss_struct{ii}.loss_valid(1, :) = loss_struct{ii}.loss_valid(1, :)/3;
%     loss_struct{ii}.loss_valid = loss_struct{ii}.loss_valid/161;
end

close all;
figure('DefaultAxesFontSize',14);
ax=subplot(2,1,1);
for ii = 1:numel(loss_struct)
    ax.ColorOrderIndex = ii;
    semilogy(epochs, loss_struct{ii}.loss_train(:,epochs),'LineStyle','--');
    hold on;
    ax.ColorOrderIndex = ii;
    semilogy(epochs, loss_struct{ii}.loss_valid(:,epochs));
end
hold off;
grid on
xlabel('epoch');
xlim([0 epochs(end)]);
ylabel('Mean squared error loss');
legend('train set', 'validation set');
% legend('frame-based (validation set, L_I)', ...
%        'frame-based (validation set, L_\alpha)');

ax = subplot(2,1,2);
for ii = 1:numel(loss_struct)
    hold on;
    ax.ColorOrderIndex = ii;
    plot(epochs, loss_struct{ii}.snr_seg_valid(:,epochs));
end
hold off;
box on;
grid on;
xlabel('epoch');
xlim([0 epochs(end)]);
ylabel('SNRseg (dB)');
legend('validation set' ,'Location','southeast');
% legend('frame-based (validation set, SNR_I)', ...
%        'frame-based (validation set, SNR_\alpha)','Location','southeast');

fig = gcf;
fname = 'loss_comp_I_a_MLP_frame_59';
set(fig,'renderer','painter');
set(fig,'Position',[50 50 1000 700]);
% print('-dpng' , '-r300' , fname)
% saveas(fig,fname,'fig')
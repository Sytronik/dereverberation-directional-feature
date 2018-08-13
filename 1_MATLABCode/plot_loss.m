% final_epoch = 60
loss_struct{1} = load('MLP_ReLU_loss_99.mat');
% loss_struct{2} = load('MLP_pReLU_loss_39.mat');
% load(['MLP_result_' num2str(final_epoch) '.mat'])
% clear 'IV_free' 'IV_room' 'IV_estimated'

epochs = 1:100
% if sum(loss_train==0)>0
%     epochs = 1:min(find(loss_train==0))-1;
% else
%     epochs = 1:length(loss_train);
% end

figure(1); clf;
ax=subplot(2,1,1);
for ii = 1:numel(loss_struct)
    ax.ColorOrderIndex = ii;
    semilogy(epochs, loss_struct{ii}.loss_train(epochs),'LineStyle','--');
    hold on;
    ax.ColorOrderIndex = ii;
    semilogy(epochs, loss_struct{ii}.loss_valid(epochs));
end
hold off;
grid on
xlabel('epoch');
xlim([0 epochs(end)]);
ylabel('Squared error loss');
legend('ReLU (train set)', 'ReLU (validation set)', ...
    'pReLU (train set)', 'pReLU (validation set)');

ax = subplot(2,1,2);
for ii = 1:numel(loss_struct)
    hold on;
    ax.ColorOrderIndex = ii;
    plot(epochs, loss_struct{ii}.snr_valid_dB(epochs));
end
hold off;
box on;
grid on;
xlabel('epoch');
xlim([0 epochs(end)]);
ylabel('SNR (dB)');
legend('ReLU (validation set)', 'pReLU(validation set)','Location','southeast');

fig = gcf;
fname = 'MLP_ReLU_loss_99';
set(fig,'renderer','painter');
set(fig,'Position',[50 50 1000 700]);
print('-dpng' , '-r300' , fname)
saveas(fig,fname,'fig')
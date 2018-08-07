% final_epoch = 60
% load(['MLP_loss_' num2str(final_epoch) '.mat'])
% load(['MLP_result_' num2str(final_epoch) '.mat'])
% clear 'IV_free' 'IV_room' 'IV_estimated'

epochs = 1:min(find(loss_train==0))-1;

figure(1); clf;
subplot(2,1,1);
plot(epochs, loss_train(epochs), epochs, loss_valid(epochs))
grid on
xlabel('epoch');
xlim([0 epochs(end)]);
ylabel('Squared error loss');
legend('train set', 'validation set');

subplot(2,1,2);
plot(epochs, snr_valid_dB(epochs));
grid on
xlabel('epoch');
xlim([0 epochs(end)]);
ylabel('SNR (dB)');
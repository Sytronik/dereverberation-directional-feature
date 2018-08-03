load('MLP_loss_49.mat')

epochs = 1:length(loss_train);

figure(1); clf;
subplot(2,1,1);
plot(epochs, loss_train, epochs, loss_valid)
grid on
xlabel('epoch');
xlim([epochs(1) epochs(end)]);
ylabel('MSE loss');
legend('train set', 'validation set');

subplot(2,1,2);
plot(epochs, snr_valid_dB);
grid on
xlabel('epoch');
xlim([epochs(1) epochs(end)]);
ylabel('SNR (dB)');
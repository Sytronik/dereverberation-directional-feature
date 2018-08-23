Fs = 16000;

[N_mic, L_RIR, N_loc] = size(RIR);
cum_reverse = cumsum(RIR(:,end:-1:1,:).^2, 2);
cum_dB = 10*log10(cum_reverse(:,end:-1:1,:));
thr_start = cum_dB(:,1,:) - 5;
thr_end = thr_start - 20;

idx_start = zeros(N_mic, N_loc);
idx_end = zeros(N_mic, N_loc);

for i_mic = 1:N_mic
    for i_loc = 1:N_loc
        idx_start(i_mic, i_loc) = find(cum_dB(i_mic, :, i_loc)<thr_start(i_mic, :, i_loc), 1);
        idx_end(i_mic, i_loc) = find(cum_dB(i_mic, :, i_loc)<thr_end(i_mic, :, i_loc), 1);
    end
end

RT20 = (idx_end - idx_start) ./ Fs;
RT60 = RT20*3;
RT60_mean = mean(RT60);

figure(1); clf;
% stem(RT60.');
stem(RT60_mean);
xlabel('Source Position Index');
ylabel('RT60 (sec)');
xlim([0 N_loc+1]);

title('Average RT60 for 32ch microphone array');

% for ii = 1:N_mic
%     lgnds{ii} = ['RT60 of mic ' num2str(ii)];
% end
% legend(lgnds);
grid;

fig = gcf;
set(fig,'renderer','painter');
% fname = 'RT60_72Azi';
% print('-dpng' , '-r300' , fname)
% saveas(fig,fname,'fig')
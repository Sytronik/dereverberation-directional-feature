Fs = 16000;

[N_mic, L_RIR, N_loc] = size(RIR);

L_win = 0.02 * Fs;
L_hop = L_win/2;
win = hamming(L_win).';
energy = zeros(N_mic, floor((L_RIR)/L_hop), N_loc);
for ii = 1:floor((L_RIR-L_win+1)/L_hop)+1
    energy(:,ii,:) = sum(RIR(:,((ii-1)*L_hop+1):(ii-1)*L_hop+L_win,:).^2 .* repmat(win, [N_mic 1 N_loc]), 2);
end

[egy_direct, idx_direct] = max(energy,[],2);
idx_direct = squeeze(idx_direct);
idx_60 = zeros(N_mic, N_loc);
for i_loc = 1:N_loc
	[i_mic, n] = find(20*log10(energy(:,:,i_loc)./egy_direct(:,:,i_loc))<=-60);
    
    for ii = 1:length(i_mic)
        if idx_60(i_mic(ii), i_loc)==0
            if n(ii) > idx_direct(i_mic(ii), i_loc)
                idx_60(i_mic(ii), i_loc) = n(ii);
            end
        end
    end
end
RT60 = (idx_60 - idx_direct)*L_hop/Fs;
RT60_mean = mean(RT60, 1);
% RT60_max = max(RT60, [], 1);

figure(1); clf;
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
fname = 'RT60_72Azi';
print('-dpng' , '-r300' , fname)
saveas(fig,fname,'fig')
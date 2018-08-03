clear;

load '0001_00.mat'

scaling = max(max(max(abs(IV_free(:,:,1:3)))));
gamma = 50
IV_free(:,:,1:3) = (IV_free(:,:,1:3) + scaling) ./ (2*scaling);
IV_free(:,:,1:3) = 1./(1+exp(-gamma*(IV_free(:,:,1:3)-0.5)));
IV_free(:,:,4) = 10*log10(IV_free(:,:,4)+1e-3);
[N_freq, x_free] = size(IV_free(:,:,1));
y = [0 8000];
x_free = [1 x_free];
scaling = max(max(max(abs(IV_room(:,:,1:3)))));
IV_room(:,:,1:3) = (IV_room(:,:,1:3) + scaling) ./ (2*scaling);
IV_room(:,:,1:3) = 1./(1+exp(-gamma*(IV_room(:,:,1:3)-0.5)));
IV_room(:,:,4) = 10*log10(IV_room(:,:,4)+1e-3);
[~, x_room] = size(IV_room(:,:,1));
x_room = [1 x_room];
x = max([x_free x_room]);

%% Plot
figure(1);clf;
ax=subplot(2,2,1);
image(x_free, y, IV_free(:,:,1:3));
ax.YDir = 'normal';
xlim([1 x]);
xlabel('frame index');
ylabel('frequency (Hz)');
title('Free-field Intensity Vector')

% hf4 = figure(4);clf;
ax=subplot(2,2,3);
image(x_free, y, IV_free(:,:,4),'CDataMapping','scaled');
ax.YDir = 'normal';
xlim([1 x]);
xlabel('frame index');
ylabel('frequency (Hz)');
colorbar
title('Free-field |a_{00}|^2 (dB)')

% hf2 = figure(2);clf;
ax=subplot(2,2,2);
image(x_room, y, IV_room(:,:,1:3));
ax.YDir = 'normal';
xlim([1 x]);
xlabel('frame index');
ylabel('frequency (Hz)');
title('Room Intensity Vector')

% hf3 = figure(3);clf;
ax=subplot(2,2,4);
image(x_room, y, IV_room(:,:,4), 'CDataMapping','scaled');
ax.YDir = 'normal';
xlim([1 x]);
xlabel('frame index');
ylabel('frequency (Hz)');
colorbar
title('Room |a_{00}|^2 (dB)')

n=3;
figure(2);clf;
ax=subplot(2,1,1);
board = repmat((checkerboard(n, floor(N_freq/2/n), floor(x/2/n))>0.5)*0.3+0.7, [1 1 3]);
image([1 x], y, board);
hold on;
image(x_free, y, IV_free(:,:,1:3), 'AlphaData', IV_free(:,:,4),'AlphaDataMapping','scaled');
ax.YDir = 'normal';
xlim([1 x]);
xlabel('frame index');
ylabel('frequency (Hz)');
title('Free-field Intensity Vector + |a_{00}|^2');
hold off;

ax=subplot(2,1,2);
image([1 x], y, board);
hold on;
image(x_room, y, IV_room(:,:,1:3), 'AlphaData', IV_room(:,:,4),'AlphaDataMapping','scaled');
ax.YDir = 'normal';
xlim([1 x]);
xlabel('frame index');
ylabel('frequency (Hz)');
title('Room Intensity Vector + |a_{00}|^2')
hold off;
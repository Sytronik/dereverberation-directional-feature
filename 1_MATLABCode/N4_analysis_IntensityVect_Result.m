clear;

load 'IV_RGBA_noreg.mat'

%% Plot
hf1 = figure(1);clf;
imagesc(flipud(permute(IV_free(:,:,1:3)/10,[2 1 3])));
xlabel('time');
ylabel('frequency');
title('Free-field Intensity vector')

hf4 = figure(4);clf;
imagesc(flipud(permute(20*log10(IV_free(:,:,4)),[2 1 3])));
xlabel('time');
ylabel('frequency');
colorbar
title('Free-field a00')

hf2 = figure(2);clf;
imagesc(flipud(permute(IV_room(:,:,1:3),[2 1 3])));
xlabel('time');
ylabel('frequency');
title('Room Intensity vector')

hf3 = figure(3);clf;
imagesc(flipud(permute(20*log10(IV_room(:,:,4)),[2 1 3])));
xlabel('time');
ylabel('frequency');
colorbar
title('Room a00')
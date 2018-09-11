if ~exist('IV_free')
    clear;
    load '0001_00.mat'
end

N_IV = 1;

IVs{N_IV} = IV_free;
titles{N_IV} = 'Free-field';

if exist('IV_0')
    N_IV = N_IV+1;
    IVs{N_IV} = IV_0;
    titles{N_IV} = 'Free-field (with Aliasing)';
end

if exist('IV_room')
    N_IV = N_IV+1;
    IVs{N_IV} = IV_room;
    titles{N_IV} = 'Reverberant';
end

if exist('IV_estimated')
    N_IV = N_IV+1;
    IVs{N_IV} = IV_estimated;
    titles{N_IV} = 'Estimated Free-field';
end



for ii = 1:N_IV
    power{ii} = IVs{ii}(:,:,4);
end

for ii = 1:N_IV
     mean_power(ii)=mean(power{ii}(:));
     disp([titles{ii} ': ' num2str(mean_power(ii))]);
end


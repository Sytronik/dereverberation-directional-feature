if ~exist('IV_free')
    clear;
    load 'MLP_result_59.mat'
end

N_IV = 1;

IVs{N_IV} = IV_free;
titles{N_IV} = 'Free-field';

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
    
%% Constants
gamma = 50;
scaling_max = 0;
x_max = 0;
for ii = 1:N_IV
    scaling{ii} = max(max(max(abs(IVs{ii}(:,:,1:3)))));
    if scaling{ii} > scaling_max
        scaling_max = scaling{ii};
    end
    
    [N_freq, temp] = size(IVs{ii}(:,:,1));
    x_axis{ii} = [1 temp];
    if temp > x_max
        x_max = temp;
    end
    
end
x_lim = [1 x_max];
y_axis = [0 8000];

%% Normalize
for ii = 1:N_IV
    % ---RGB---
    % The same scaling factor
    scaling{ii}=scaling_max;

    % scaling
    IVs{ii}(:,:,1:3) = (IVs{ii}(:,:,1:3) + scaling{ii}) ./ (2*scaling{ii});
    
    % contrast enhancment
    IVs{ii}(:,:,1:3) = 1./(1+exp(-gamma*(IVs{ii}(:,:,1:3)-0.5)));
%     IVs{ii}(:,:,1:3) = (IVs{ii}(:,:,1:3)-0.5)/2./max(max(max(abs(IVs{ii}(:,:,1:3)-0.5))))+0.5;

    % ---alpha---
    % clip negative values
    IVs{ii}(:,:,4) = (IVs{ii}(:,:,4)>0).*IVs{ii}(:,:,4);
    
    % dB scale
    IVs{ii}(:,:,4) = 10*log10(IVs{ii}(:,:,4)+1e-3);
end

%% Colormap of alpha
c_max = -inf;
c_min = inf;
for ii = 1:N_IV
    temp = max(max(IVs{ii}(:,:,4)));
    if temp > c_max
        c_max = temp;
    end
    
    temp = min(min(IVs{ii}(:,:,4)));
    if temp < c_min
        c_min = temp;
    end
    
end
c_lim = [c_min c_max];

%% Plot
figure(1);clf;
for ii = 1:N_IV
    ax=subplot(2, N_IV, ii);
    image(x_axis{ii}, y_axis, IVs{ii}(:,:,1:3));
    ax.YDir = 'normal';
    xlim(x_lim);
    xlabel('frame index');
    ylabel('frequency (Hz)');
%     title([titles{ii} ' Intensity Vector $\tilde\mathbf{I}(\tau,f)$'], 'Interpreter', 'latex')
    title([titles{ii} ' Intensity Vector $\hat\mathbf{I}(\tau,f)$'], 'Interpreter', 'latex')

    ax=subplot(2, N_IV, N_IV + ii);
    image(x_axis{ii}, y_axis, IVs{ii}(:,:,4),'CDataMapping','scaled');
    ax.YDir = 'normal';
    xlim(x_lim);
    ax.CLim=c_lim;
    xlabel('frame index');
    ylabel('frequency (Hz)');
    colorbar
    title([titles{ii} ' $|a_{00}(\tau,f)|^2$ (dB)'], 'Interpreter', 'latex')
end

n=2;
figure(2);clf;
for ii = 1:N_IV
    ax=subplot(N_IV,1,ii);
    board = repmat((checkerboard(n, ceil(N_freq/2/n), ceil(x_max/2/n))>0.5)*0.2+0.8, [1 1 3]);
    board = board(1:N_freq, 1:x_max, :);
    image(x_lim, y_axis, board);
    hold on;
    image(x_axis{ii}, y_axis, IVs{ii}(:,:,1:3), ...
          'AlphaData', IVs{ii}(:,:,4), 'AlphaDataMapping', 'scaled');
    ax.YDir = 'normal';
    xlim(x_lim);
    xlabel('frame index');
    ylabel('frequency (Hz)');
    title([titles{ii} ' $\hat\mathbf{I}(\tau,f)$ \& $|a_{00}(\tau,f)|^2$ (dB)'], 'Interpreter', 'latex');
    hold off;
end
clear;
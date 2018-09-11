if ~exist('IV_free')
    clear;
    load 'MLP_frame_result_59_test.mat'
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


    
%% Constants
gamma = 1/7;
x_max = 0;
for ii = 1:N_IV
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
    % spherical coordinate
    [azi, elv, r] = cart2sph(IVs{ii}(:,:,1), IVs{ii}(:,:,2), IVs{ii}(:,:,3));
    azi = (azi<0).*(2*pi+azi) + (azi>=0).*azi;  % 0~2pi
    elv = pi/2 - elv;  % Standard
    
    % normalize
    azi = azi / (2*pi);
    elv = elv / pi;
    r = r/max(r(:));
    
    % HSV->RGB
    hsv = zeros(size(IVs{ii}(:,:,1:3)));
    hsv(:,:,1) = azi;
    hsv(:,:,2) = elv;
    hsv(:,:,3) = r;
    IVs{ii}(:,:,1:3) = hsv2rgb(hsv);
    
    % contrast enhancment
    IVs{ii}(:,:,1:3) = IVs{ii}(:,:,1:3).^gamma;
%     IVs{ii}(:,:,1:3) = 1./(1+exp(-gamma*(IVs{ii}(:,:,1:3)-0.5)));
%     IVs{ii}(:,:,1:3) = (IVs{ii}(:,:,1:3)-0.5)/2./max(max(max(abs(IVs{ii}(:,:,1:3)-0.5))))+0.5;

    % ---alpha---
    % clip negative values
    IVs{ii}(:,:,4) = (IVs{ii}(:,:,4)>0).*IVs{ii}(:,:,4);
    
    % normalize
    IVs{ii}(:,:,4) = IVs{ii}(:,:,4) / max(max(IVs{ii}(:,:,4)));
    
    % dB scale
    IVs{ii}(:,:,4) = 10*log10(IVs{ii}(:,:,4)+1e-4);
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
close all;
figure('DefaultAxesFontSize',14, 'Position',[50 40 1600 950]);
fig = gcf;
set(fig,'renderer','painter');
for ii = 1:N_IV
    ax=subplot(N_IV, 2, 2*ii-1);
    image(x_axis{ii}, y_axis, IVs{ii}(:,:,1:3));
    ax.YDir = 'normal';
    xlim(x_lim);
    xlabel('frame index');
    ylabel('frequency (Hz)');
%     title([titles{ii} ' Intensity Vector $\tilde\mathbf{I}(\tau,f)$'], 'Interpreter', 'latex')
    title([titles{ii} ' Intensity Vector $\mathbf{I}(\tau,f)$ (HSV)'], 'Interpreter', 'latex')
    c = colorbar('east');
    drawnow;
    c.Location = 'eastoutside';
    c.Visible = 0;
    
    ax=subplot(N_IV, 2, 2*ii);
    image(x_axis{ii}, y_axis, IVs{ii}(:,:,4),'CDataMapping','scaled');
    ax.YDir = 'normal';
    xlim(x_lim);
    ax.CLim=c_lim;
    xlabel('frame index');
    ylabel('frequency (Hz)');
    title([titles{ii} ' Power Spectrum (dB)'])
    c = colorbar('east');
    drawnow;
    c.Location = 'eastoutside';
end
% fname = ['MLP_frame_result_23_' num2str(N_IV)];
% print('-dpng' , '-r600' , fname)
% saveas(fig,fname,'fig')

n=2;
figure('DefaultAxesFontSize',14, 'Position',[50 40 800 950]);
fig = gcf;
set(fig,'renderer','painter');
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
    title([titles{ii} ' $\mathbf{I}(\tau,f)$ (HSV) \& Power Spectrum (dB)'], 'Interpreter', 'latex');
    hold off;
end
% fname = ['MLP_frame_result_23_merge_a_' num2str(N_IV)];
% print('-dpng' , '-r600' , fname)
% saveas(fig,fname,'fig')
% clear;
% close all;
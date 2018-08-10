raw_struct{1} = load('MLP_ReLU_39.mat', '*weight');
raw_struct{2} = load('MLP_pReLU_39.mat', '*weight');
titles{1} = 'Histogram of Weights (ReLU)';
titles{2} = 'Histogram of Weights (pReLU)';

weights = cell(size(raw_struct));

for ii = 1:numel(raw_struct)
    weights{ii} = [];
    names = fieldnames(raw_struct{ii});
    for jj = 1:numel(names)
        data = raw_struct{ii}.(names{jj});
        if numel(data)~=1
            weights{ii} = [weights{ii};data(:)];
        end
    end
end

%% Comparision
figure(1);
fig1 = gcf;
for ii = 1:numel(raw_struct)
    subplot(1,2,ii);
    ax = histogram(weights{ii})
    xlabel('Neural Network Weights')
    ylabel('frequency')
    title(titles{ii})
    grid on
    xlim([-1.5 1.5])
    ylim([0 7.1e7])
end
fname1 = 'hist_w_comp_ReLU_pReLU';
set(fig1,'renderer','painter');
set(fig1,'Position',[50 50 1500 500]);
print('-dpng' , '-r300' , fname1)
% saveas(fig1,fname1,'fig')

%% pReLU
figure(2);
fig2 = gcf;
histogram(weights{2})
xlabel('Neural Network Weights')
ylabel('frequency')
title(titles{2})
grid on
fname2 = 'hist_w_pReLU';
set(fig2,'renderer','painter');
set(fig2,'Position',[50 50 750 500]);
print('-dpng' , '-r300' , fname2)
% saveas(fig2,fname2,'fig')
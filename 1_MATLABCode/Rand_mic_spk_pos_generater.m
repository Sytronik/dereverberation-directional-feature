close all
clear

dist3D = @(x1,y1,z1,x2,y2,z2) sqrt((x1-x2)^2+(y1-y2)^2+(z1-z2)^2);

%%  Room dimension
L = [10.52 7.1 3.02];                   % Room dimensions (x,y,z) in m

%% random position generation
Nsel = 100;
Nrand = 1000;
Rand_gen_L_s = [rand(Nrand,1)*L(1) rand(Nrand,1)*L(2) rand(Nrand,1)*L(3)];
Rand_gen_L_m = [rand(Nrand,1)*L(1) rand(Nrand,1)*L(2) rand(Nrand,1)*L(3)];


%% Ban special situation
% exception criteria
% minimum wall distance : 0.5 m
% minimum distance between spk and mic position : 0.5 m
Spk_result = cell(Nsel,1);
Mic_result = cell(Nsel,1);
cnt = 0;
cnt2 = 0;

while cnt2 < Nsel
    
    cnt = cnt + 1;
    
    Condx1 = (Rand_gen_L_s(cnt,1)>=0.5);
    Condx2 = (Rand_gen_L_s(cnt,1)<=L(1)-0.5);
    
    Condy1 = (Rand_gen_L_s(cnt,2)>=0.5);
    Condy2 = (Rand_gen_L_s(cnt,2)<=L(2)-0.5);
    
    Condz1 = (Rand_gen_L_s(cnt,3)>=0.5);
    Condz2 = (Rand_gen_L_s(cnt,3)<=L(3)-0.5);
    
    Condx1m = (Rand_gen_L_m(cnt,1)>=0.5);
    Condx2m = (Rand_gen_L_m(cnt,1)<=L(1)-0.5);
    
    Condy1m = (Rand_gen_L_m(cnt,2)>=0.5);
    Condy2m = (Rand_gen_L_m(cnt,2)<=L(2)-0.5);
    
    Condz1m = (Rand_gen_L_m(cnt,3)>=0.5);
    Condz2m = (Rand_gen_L_m(cnt,3)<=L(3)-0.5);
    
    Cond_spk_mic_dist = dist3D(Rand_gen_L_s(cnt,1),Rand_gen_L_s(cnt,2),Rand_gen_L_s(cnt,3),...
                                Rand_gen_L_m(cnt,1),Rand_gen_L_m(cnt,2),Rand_gen_L_m(cnt,3))>=0.5;
    
    if  (Condx1 == 1) && (Condx2 == 1) && ...
        (Condy1 == 1) && (Condy2 == 1) && ...
        (Condz1 == 1) && (Condz2 == 1) && ...
        (Condx1m == 1) && (Condx2m == 1) && ...
        (Condy1m == 1) && (Condy2m == 1) && ...
        (Condz1m == 1) && (Condz2m == 1) && Cond_spk_mic_dist
    
        cnt2 = cnt2 + 1;
        Spk_result{cnt2} = Rand_gen_L_s(cnt,:);
        Mic_result{cnt2} = Rand_gen_L_m(cnt,:);
    
    end
       
end

% validation 
figure(1)
for ii = 1:5
hcs_spk = scatter3(Spk_result{ii}(1),Spk_result{ii}(2),Spk_result{ii}(3));
hold on;
hcs_mic = scatter3(Mic_result{ii}(1),Mic_result{ii}(2),Mic_result{ii}(3));
set(hcs_spk,'Marker','diamond','MarkerFaceColor',rand(1,3),'LineWidth',1.5);
set(hcs_mic,'Marker','o','MarkerFaceColor',rand(1,3),'LineWidth',1.5);

end

grid on;
title('spk(red), mic(blue) result');



%% speaker position, microphone array position matching
s           = Spk_result;                    % Source location(s) (x,y,z) in m
sphLocation = Mic_result;       % Receiver location (x,y,z) in m

%% save
save('Uniformly_random_spk_mic_pos.mat','s','sphLocation');

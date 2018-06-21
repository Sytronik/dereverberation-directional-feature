clear
close all

addpath SMIR-Generator-master  myfun polarch-Spherical-Array-Processing-4df5f23 speech

%% source signal
[src_signal,Fs_signal] = audioread('ar-01.wav');

%% Spherical microphone FIR filtering
RIR_result = load('Anm_FIR_6order.mat');
IR_SH_mic = RIR_result.h1;

IR_FIR = cell(1,32);
for ii = 1:32
    IR_FIR{ii} = dsp.FIRFilter('Numerator',IR_SH_mic(ii,:));
    FilteredSignal(:,ii) = step(IR_FIR{ii},src_signal(:));
    ii
end

save('Filtered_Speech_6order.mat') 
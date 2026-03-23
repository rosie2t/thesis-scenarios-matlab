% Scenario 1: 5G Channel Estimation using CNN
clear; clc; close all;

% Simulation Parameters
nSubcarriers = 72;    % Bandwidth
nSymbols = 14;        % Number of Symbols
snr_dB = 10;          % Noise level

% Different noise levels to try
%snr_dB = -5;
%snr_dB = 20;          

% To try mobility just un comment the code block
%{
%for mobility
fc = 3.5e9;               %frequency
v = 3 / 3.6;              %velocity of user => changing this field shows how ai is better
c = 3e8;                 
dopplerShift = (v * fc) / c; %doppler shift

%in channel statistcs
channel.MaximumDopplerShift = dopplerShift;

% Pssing the signal through the channel => adding samples for this scenario
nSamples = nSubcarriers * nSymbols;
[out, pathGains] = channel(ones(nSamples, 1));
H_actual_raw = fft(pagetranspose(pathGains), nSubcarriers);
H_actual = H_actual_raw(:, 1:nSymbols);
%}
% Different velocities for mobility
% v = 3 / 3.6;            
% v = 60 / 3.6;
% v = 120 / 3.6;             

% Making a Resource Grid
resourceGrid = complex(randn(nSubcarriers, nSymbols), randn(nSubcarriers, nSymbols));

% Channel modeling (TDL)
% Using a standard 5G channel TDL-C
channel = nrTDLChannel;
channel.DelayProfile = 'TDL-C';
channel.DelaySpread = 300e-9; % Delay 300ns
channel.SampleRate = 15.36e6;

% Pssing the signal through the channel
[out, pathGains] = channel(ones(1000, 1));
H_actual = fft(pathGains(1, :), nSubcarriers).';
H_actual = repmat(H_actual, 1, nSymbols);

% Signal Reception and Traditional Estimation
% adding a channel and noise
rxSignal = H_actual .* resourceGrid;
rxSignalNoisy = awgn(rxSignal, snr_dB, 'measured');

% Traditional Estimation: Least Squares (LS) => H_est = Rx / Tx 
H_LS = rxSignalNoisy ./ resourceGrid;

% Preparing data for CNN
% CNN does not work with complex numbers, so I break it down into real and imaginary
% Creating a two layer image, using real and imaginary 
X_input = cat(3, real(H_LS), imag(H_LS));
Y_target = cat(3, real(H_actual), imag(H_actual));

% CNN Architecture
% Creating the learning layers
layers = [
   imageInputLayer([nSubcarriers nSymbols 2], 'Name', 'Input') % Input: The noisy channel
  
   convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'Conv1') % 1st learning level
   reluLayer('Name', 'ReLU1')                                   % Activate
  
   convolution2dLayer(3, 8, 'Padding', 'same', 'Name', 'Conv2')  % 2nd learning layer
   reluLayer('Name', 'ReLU2')
  
   convolution2dLayer(3, 2, 'Padding', 'same', 'Name', 'OutputConv') % Output 2-levels (Real/Imag)
   regressionLayer('Name', 'Output')
];

% Training and Prediction
options = trainingOptions('adam', ...
   'MaxEpochs', 50, ...
   'InitialLearnRate', 0.01, ...
   'Plots', 'training-progress', ...
   'Verbose', false);

% Training the network
net = trainNetwork(X_input, Y_target, layers, options);

% Predictions from CNN
H_AI_raw = predict(net, X_input);
H_AI = H_AI_raw(:,:,1) + 1i*H_AI_raw(:,:,2); 

% Comparing and showing the results
mse_LS = mean(abs(H_actual - H_LS).^2, 'all');
mse_AI = mean(abs(H_actual - H_AI).^2, 'all');
fprintf('Σφάλμα (MSE) Παραδοσιακής Μεθόδου (LS): %f\n', mse_LS);
fprintf('Σφάλμα (MSE) Τεχνητής Νοημοσύνης (CNN): %f\n', mse_AI);
figure;
subplot(1,3,1); imagesc(abs(H_actual)); title('Πραγματικό Κανάλι'); colorbar;
subplot(1,3,2); imagesc(abs(H_LS)); title('Εκτίμηση LS (Με θόρυβο)'); colorbar;
subplot(1,3,3); imagesc(abs(H_AI)); title('Εκτίμηση CNN (AI)'); colorbar;
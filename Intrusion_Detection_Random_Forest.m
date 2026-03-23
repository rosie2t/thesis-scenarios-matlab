% Scenario 4: Intrusion Detection using Random Forest
clear; clc;

% Creating Data (Features: PacketSize, InterArrival, Flags)
numSamples = 1000;

% Normal Traffic (Low Rate, Small Packets)
normalData = [normrnd(500, 50, [numSamples/2, 1]), normrnd(0.1, 0.02, [numSamples/2, 1])];

% DDoS attack (High rate - small InterArrival, Many packets)
attackData = [normrnd(800, 100, [numSamples/2, 1]), normrnd(0.01, 0.005, [numSamples/2, 1])];
X = [normalData; attackData];
Y = [zeros(numSamples/2, 1); ones(numSamples/2, 1)]; % 0: Normal, 1: Attack
%for more realistic attack packets
% Y = [zeros(numSamples * 0.95, 1); ones(numSamples * 0.05, 1)];

% Data shuffling
idx = randperm(numSamples);
X = X(idx, :); Y = Y(idx);

% Training Random Forest
numTrees = 50;
model = fitcensemble(X, Y, 'Method', 'Bag', 'NumLearningCycles', numTrees, 'Learners', 'tree');

% Prediction and Evaluation
yPred = predict(model, X);
accuracy = sum(yPred == Y) / numSamples * 100;
fprintf('Accuracy of Intrusion Detection: %.2f%%\n', accuracy);

% Showing results
figure;
gscatter(X(:,1), X(:,2), Y, 'rb', 'ox');
title('Ανίχνευση Εισβολών στο 5G Network Slice');
xlabel('Packet Size (Bytes)'); ylabel('Inter-arrival Time (sec)');
legend('Normal', 'Attack'); grid on;

%confusion matrix
figure;
confusionchart(Y, yPred, 'Title', 'Πίνακας Σύγχυσης: Ανίχνευση Εισβολών');
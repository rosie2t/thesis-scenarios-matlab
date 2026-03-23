% Scenario 5: Beamforming Optimization using DNN
clear; clc;

% Creating Data (Input: Position X,Y => Output: Ideal Angle)
numUsers = 2000;
posX = rand(numUsers, 1) * 100; % users within a radius of 100m
posY = rand(numUsers, 1) * 100;

% The ideal angle is calculated geometrically (ground truth)
idealAngle = atan2d(posY, posX);
X = [posX, posY];
Y = idealAngle;

% Structure of Deep Neural Network (Regression)
layers = [
   featureInputLayer(2, 'Name', 'input')
   fullyConnectedLayer(64, 'Name', 'fc1')
   reluLayer('Name', 'relu1')
   fullyConnectedLayer(32, 'Name', 'fc2')
   reluLayer('Name', 'relu2')
   fullyConnectedLayer(1, 'Name', 'output')
   regressionLayer('Name', 'loss')];

% Training parameters
options = trainingOptions('adam', ...
   'MaxEpochs', 100, ...
   'MiniBatchSize', 32, ...
   'InitialLearnRate', 0.01, ...
   'Plots', 'training-progress', ...
   'Verbose', false);

% Training the network
net = trainNetwork(X, Y, layers, options);

% Testing with a new user
testUserPos = [30, 70]; % position of user
predictedAngle = predict(net, testUserPos);
%to predict all the dataset
%predictedAngles = predict(net, X);
%error = Y - predictedAngles;
fprintf('Για χρήστη στη θέση (30,70), η προβλεπόμενη γωνία Beamforming είναι: %.2f μοίρες\n', predictedAngle);

%{
% results for all the dataset -> error prediction
figure;
histogram(error, 30); %error
title('Κατανομή Σφάλματος Πρόβλεψης Γωνίας Beamforming');
xlabel('Σφάλμα (Μοίρες)'); ylabel('Πλήθος Χρηστών');
grid on;
%}
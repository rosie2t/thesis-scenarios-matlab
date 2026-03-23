% Scenario 3: Path Loss Prediction using ML
clear; clc; close all;
% Creating Data
dist = (10:10:1000)';          % Distances (from 10m to 1km)
f = 2.4e9;                     % Frequency (2.4 GHz)
c = 3e8;                       % Speed of light
% Theoretical model (Free Space Path Loss) as a basis
PL_theoretical = 20*log10(dist) + 20*log10(f) - 147.55;
%PL_theoretical = 25*log10(dist) + 20*log10(f) - 147.55;

% Adding real conditions with random noise
PL_measured = PL_theoretical + normrnd(0, 5, length(dist), 1);
%PL_measured = PL_theoretical + normrnd(0, 8, length(dist), 1);
%PL_measured = PL_theoretical + normrnd(0, 15, length(dist), 1);

% ML Model Training
% Using SVR to learn the pattern of loss
fprintf('Εκπαίδευση μοντέλου Machine Learning...\n');

% Training (using the distances to predict the pattern loss)
mdl = fitrsvm(dist, PL_measured, 'Standardize', true, 'KernelFunction', 'gaussian', 'OptimizeHyperparameters', 'auto');

% Prediction
PL_AI_pred = predict(mdl, dist);

% Comparison and Results
% Calculating error (RMSE)
rmse_traditional = sqrt(mean((PL_measured - PL_theoretical).^2));
rmse_AI = sqrt(mean((PL_measured - PL_AI_pred).^2));
fprintf('Σφάλμα (RMSE) Παραδοσιακού Μοντέλου: %.2f dB\n', rmse_traditional);
fprintf('Σφάλμα (RMSE) AI Μοντέλου: %.2f dB\n', rmse_AI);

% Showing Results
figure;
scatter(dist, PL_measured, 'k', 'displayName', 'Πραγματικές Μετρήσεις'); hold on;
plot(dist, PL_theoretical, 'r--', 'LineWidth', 2, 'displayName', 'Θεωρητικό Μοντέλο');
plot(dist, PL_AI_pred, 'b-', 'LineWidth', 2, 'displayName', 'AI Πρόβλεψη (SVR)');
grid on;
xlabel('Απόσταση (m)');
ylabel('Path Loss (dB)');
legend('Location', 'southeast');
title('Πρόβλεψη Απώλειας Σήματος: AI vs Θεωρητικό Μοντέλο');
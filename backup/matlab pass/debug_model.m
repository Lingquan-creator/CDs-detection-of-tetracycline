% debug_model.m - Debug the model to see what's happening

clear; clc;

% Read data
data = readmatrix('yu 2.csv');
data = data(22:end, :);
lambda_exp = data(:, 1);
I_exp = data(:, 2);
valid_idx = isfinite(lambda_exp) & isfinite(I_exp);
lambda_exp = lambda_exp(valid_idx);
I_exp = I_exp(valid_idx);

% Constants
C = 1239.8;
Ap = 4.87e10;  % From working model
lambda0 = 448.15;
sigma_E = 0.1845;

% Test the E0 calculation
E0_v1 = C / lambda0;  % Simple version
E0_v2 = C / lambda0 - 2*sigma_E^2/C * lambda0;  % With correction

fprintf('E0 simple: %.4f eV\n', E0_v1);
fprintf('E0 with correction: %.4f eV\n', E0_v2);
fprintf('Correction term: %.4f eV\n', -2*sigma_E^2/C * lambda0);

% Test the spectrum at peak wavelength
lambda_test = lambda0;
E_test = C / lambda_test;

% Version 1: simple
I_v1 = (Ap / lambda_test^2) * exp(-((E_test - E0_v1)^2) / (2*sigma_E^2));

% Version 2: with correction  
I_v2 = (Ap / lambda_test^2) * exp(-((E_test - E0_v2)^2) / (2*sigma_E^2));

fprintf('\nAt peak wavelength λ=%g nm:\n', lambda_test);
fprintf('E_photon = C/λ = %.4f eV\n', E_test);
fprintf('I_v1 (simple) = %.4e\n', I_v1);
fprintf('I_v2 (with correction) = %.4e\n', I_v2);

% Calculate spectrum over full range
I_spectrum_v1 = zeros(size(lambda_exp));
I_spectrum_v2 = zeros(size(lambda_exp));

for i = 1:length(lambda_exp)
    E_i = C / lambda_exp(i);
    I_spectrum_v1(i) = (Ap / lambda_exp(i)^2) * exp(-((E_i - E0_v1)^2) / (2*sigma_E^2));
    I_spectrum_v2(i) = (Ap / lambda_exp(i)^2) * exp(-((E_i - E0_v2)^2) / (2*sigma_E^2));
end

fprintf('\nSpectrum stats:\n');
fprintf('V1 - max: %.4e, at λ=%.1f nm\n', max(I_spectrum_v1), lambda_exp(I_spectrum_v1 == max(I_spectrum_v1)));
fprintf('V2 - max: %.4e, at λ=%.1f nm\n', max(I_spectrum_v2), lambda_exp(I_spectrum_v2 == max(I_spectrum_v2)));
fprintf('Exp data - max: %.4e, at λ=%.1f nm\n', max(I_exp), lambda_exp(I_exp == max(I_exp)));

figure;
subplot(211);
plot(lambda_exp, I_exp, 'k-', 'LineWidth', 1.5); hold on;
plot(lambda_exp, I_spectrum_v1, 'r-', 'LineWidth', 1.5);
legend('Experimental', 'Model v1 (simple)');
ylabel('Intensity');
title('Model v1: Simple E_0 = C/\lambda_0');
grid on;

subplot(212);
plot(lambda_exp, I_exp, 'k-', 'LineWidth', 1.5); hold on;
plot(lambda_exp, I_spectrum_v2, 'b-', 'LineWidth', 1.5);
legend('Experimental', 'Model v2 (with correction)');
xlabel('Wavelength (nm)');
ylabel('Intensity');
title('Model v2: E_0 = C/\lambda_0 - 2\sigma^2_{E}/C \cdot \lambda_0');
grid on;

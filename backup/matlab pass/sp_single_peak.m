% sp_single_peak.m - 量子点光谱拟合（单峰模型，k_m = 0）
%
% 根据公式：
% E_{0}(A') = C/λ_0 - 2σ²_E/C · λ_0
% I(λ; A') = (A'/λ²) * exp[-(C/λ - E_0)² / (2σ²_E)]
% S(λ) = I(λ; A') + B

clear; clc; close all;

%% 1. 读取数据
data = readmatrix('yu 2.csv');
% 跳过前21行的头部信息
data = data(22:end, :);
% 第一列是波长，第二列是强度数据
lambda_exp = data(:, 1);
I_exp = data(:, 2);

% 移除NaN和Inf值
valid_idx = isfinite(lambda_exp) & isfinite(I_exp);
lambda_exp = lambda_exp(valid_idx);
I_exp = I_exp(valid_idx);

fprintf('数据范围: λ = %.1f ~ %.1f nm, I = %.2e ~ %.2e\n', ...
    min(lambda_exp), max(lambda_exp), min(I_exp), max(I_exp));

%% 2. 定义模型参数
C = 1239.8;  % eV·nm (光子能量-波长转换常数)

%% 3. 估计峰位置和初始参数
[pks, locs] = findpeaks(I_exp, lambda_exp, 'NPeaks', 1, 'SortStr', 'descend');
if isempty(locs)
    % 如果找不到峰，使用最大值位置
    [~, max_idx] = max(I_exp);
    locs = lambda_exp(max_idx);
    pks = I_exp(max_idx);
end

B0 = min(I_exp);  % 背景估计

fprintf('\n检测到的峰位置: λ ≈ %.1f nm\n', locs);

%% 4. 初始参数设置
% 参数: [A', lambda0, sigma_E, B]
% A' 需要根据峰高度估计
% 在峰位处: I(lambda0) ≈ A' / lambda0^2
A0 = (pks - B0) * locs^2;  % 反推A'
lambda0 = locs;
sigma_E0 = 0.12;  % eV
params0 = [A0, lambda0, sigma_E0, B0];

fprintf('\n初始参数:\n');
fprintf('  A'' = %.4e\n', A0);
fprintf('  λ_0 = %.2f nm\n', lambda0);
fprintf('  σ_E = %.4f eV\n', sigma_E0);
fprintf('  B = %.4e\n', B0);

%% 5. 参数约束
lb = [0, min(lambda_exp), 0.01, 0];  % A'>=0, lambda0>=min, sigma_E>=0.01, B>=0
ub = [Inf, max(lambda_exp), 0.5, max(I_exp)*0.5];  % sigma_E<=0.5 eV, B<=50% max

%% 6. 定义模型函数 (单峰, k_m = 0)
model_func = @(params, lambda) single_peak_spectrum(lambda, params, C);

%% 7. 非线性拟合
options = optimoptions('lsqcurvefit', ...
    'Display', 'iter', ...
    'MaxIterations', 2000, ...
    'MaxFunctionEvaluations', 20000, ...
    'FunctionTolerance', 1e-8, ...
    'StepTolerance', 1e-8);

fprintf('\n开始拟合...\n');
params_fit = lsqcurvefit(model_func, params0, lambda_exp, I_exp, lb, ub, options);

%% 8. 计算拟合结果
I_fit = model_func(params_fit, lambda_exp);

%% 9. 输出拟合参数
Ap_fit = params_fit(1);
lambda0_fit = params_fit(2);
sigma_E_fit = params_fit(3);
B_fit = params_fit(4);

fprintf('\n===== 拟合参数 =====\n');
fprintf('  A'' = %.4e\n', Ap_fit);
fprintf('  λ_0 = %.2f nm\n', lambda0_fit);
fprintf('  σ_E = %.4f eV\n', sigma_E_fit);
fprintf('  峰位 = %.2f nm (k=0时固定)\n', lambda0_fit);
fprintf('  背景 B = %.4e\n', B_fit);

% 计算峰的积分强度（不含背景）
I_peak_only = compute_peak(lambda_exp, Ap_fit, lambda0_fit, sigma_E_fit, C);
peak_integral = trapz(lambda_exp, I_peak_only);
fprintf('  积分强度 = %.4e\n', peak_integral);

% 计算中心能量
E0 = C / lambda0_fit - 2*sigma_E_fit^2/C * lambda0_fit;
fprintf('  中心能量 E_0 = %.4f eV\n', E0);

%% 10. 拟合质量
residuals = I_exp - I_fit;
SS_res = sum(residuals.^2);
SS_tot = sum((I_exp - mean(I_exp)).^2);
R_squared = 1 - SS_res / SS_tot;
RMSE = sqrt(mean(residuals.^2));

fprintf('\n===== 拟合质量 =====\n');
fprintf('R² = %.6f\n', R_squared);
fprintf('RMSE = %.4e\n', RMSE);
fprintf('相对RMSE = %.2f%%\n', 100 * RMSE / mean(I_exp));

%% 11. 绘图
figure('Position', [100, 100, 1200, 900]);

% 主图：实验数据vs拟合
subplot(3,1,1);
plot(lambda_exp, I_exp, 'k.', 'MarkerSize', 4, 'DisplayName', '实验数据');
hold on;
plot(lambda_exp, I_fit, 'r-', 'LineWidth', 2, 'DisplayName', '总拟合');

% 绘制峰（不含背景）
plot(lambda_exp, I_peak_only, 'b--', 'LineWidth', 1.5, ...
    'DisplayName', sprintf('单峰 (%.1f nm)', lambda0_fit));

% 绘制背景
plot(lambda_exp, ones(size(lambda_exp))*B_fit, 'k:', 'LineWidth', 1, 'DisplayName', '背景');

xlabel('波长 (nm)');
ylabel('强度 (a.u.)');
legend('Location', 'best');
title(sprintf('量子点单峰光谱拟合 (R^2 = %.6f)', R_squared));
grid on;
set(gca, 'FontSize', 11);

% 残差图
subplot(3,1,2);
plot(lambda_exp, residuals, 'b-', 'LineWidth', 1);
hold on;
plot(lambda_exp, zeros(size(lambda_exp)), 'k--');
xlabel('波长 (nm)');
ylabel('残差 (a.u.)');
title('拟合残差');
grid on;
set(gca, 'FontSize', 11);

% 相对残差图
subplot(3,1,3);
relative_residuals = 100 * residuals ./ I_exp;
plot(lambda_exp, relative_residuals, 'g-', 'LineWidth', 1);
hold on;
plot(lambda_exp, zeros(size(lambda_exp)), 'k--');
xlabel('波长 (nm)');
ylabel('相对残差 (%)');
title('相对拟合残差');
grid on;
set(gca, 'FontSize', 11);

%% 12. 输出拟合结果到文件
results_file = 'fit_results_single_peak.txt';
fid = fopen(results_file, 'w');
fprintf(fid, '===== 单峰拟合结果 =====\n\n');
fprintf(fid, '拟合参数:\n');
fprintf(fid, '  A'' = %.6e\n', Ap_fit);
fprintf(fid, '  λ_0 = %.4f nm\n', lambda0_fit);
fprintf(fid, '  σ_E = %.6f eV\n', sigma_E_fit);
fprintf(fid, '  背景 B = %.6e\n', B_fit);
fprintf(fid, '  中心能量 E_0 = %.6f eV\n', E0);
fprintf(fid, '  积分强度 = %.6e\n\n', peak_integral);
fprintf(fid, '拟合质量:\n');
fprintf(fid, '  R² = %.8f\n', R_squared);
fprintf(fid, '  RMSE = %.6e\n', RMSE);
fprintf(fid, '  相对RMSE = %.4f%%\n', 100 * RMSE / mean(I_exp));
fclose(fid);
fprintf('\n结果已保存到 %s\n', results_file);

%% 辅助函数：计算单峰光谱
function S = single_peak_spectrum(lambda, params, C)
    Ap = params(1);
    lambda0 = params(2);
    sigma_E = params(3);
    B = params(4);
    
    % 计算中心能量 E_0
    E0 = C / lambda0 - 2*sigma_E^2/C * lambda0;
    
    % 单峰谱形
    I_peak = (Ap ./ lambda.^2) .* exp(-((C./lambda - E0).^2) ./ (2*sigma_E^2));
    
    % 总谱 = 峰 + 背景
    S = I_peak + B;
end

%% 辅助函数：计算单峰（不含背景）
function I = compute_peak(lambda, Ap, lambda0, sigma_E, C)
    E0 = C / lambda0 - 2*sigma_E^2/C * lambda0;
    I = (Ap ./ lambda.^2) .* exp(-((C./lambda - E0).^2) ./ (2*sigma_E^2));
end

% sp_fixed.m - 量子点光谱拟合（严格模型，峰位-强度耦合）

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
n_peaks = 3; % 峰数量

%% 3. 估计峰位置和初始参数
[pks, locs] = findpeaks(I_exp, lambda_exp, 'NPeaks', n_peaks, 'SortStr', 'descend');
if length(locs) < n_peaks
    locs = linspace(min(lambda_exp)+50, max(lambda_exp)-50, n_peaks);
end
locs = sort(locs);  % 按波长排序

B0 = min(I_exp);  % 背景估计

fprintf('\n检测到的峰位置:\n');
for i = 1:n_peaks
    fprintf('  峰 %d: λ ≈ %.1f nm\n', i, locs(i));
end

%% 4. 简化模型：假设峰位不随强度变化 (k_m = 0)
% 参数: [A'_1, lambda0_1, sigma_E1, ..., A'_n, lambda0_n, sigma_En, B]
params0 = [];
for i = 1:n_peaks
    % 对于 k_m = 0 的情况，峰位固定在 lambda0_m
    % A'_m 需要根据峰高度估计
    % 在峰位处: I(lambda0) ≈ A' / lambda0^2
    if i <= length(pks)
        A0 = (pks(i) - B0) * locs(i)^2;  % 反推A'
    else
        A0 = (max(I_exp) - B0) * locs(i)^2 / n_peaks;
    end
    lambda0 = locs(i);
    sigma_E0 = 0.12;  % eV
    params0 = [params0, A0, lambda0, sigma_E0];
end
params0 = [params0, B0];

%% 5. 参数约束
lb = [];
ub = [];
for i = 1:n_peaks
    lb = [lb, 0, min(lambda_exp), 0.01];  % A'>=0, lambda0>=min, sigma_E>=0.01
    ub = [ub, Inf, max(lambda_exp), 0.5];  % sigma_E<=0.5 eV
end
lb = [lb, 0];  % B >= 0
ub = [ub, max(I_exp)*0.5];  % B <= 50% of max

%% 6. 定义模型函数 (k_m = 0)
model_func = @(params, lambda) model_spectrum_simple(lambda, params, n_peaks, C);

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
fprintf('\n===== 拟合参数 =====\n');
for i = 1:n_peaks
    idx = (i-1)*3 + 1;
    Ap_fit = params_fit(idx);
    lambda0_fit = params_fit(idx+1);
    sigma_E_fit = params_fit(idx+2);
    
    fprintf('峰 %d:\n', i);
    fprintf('  A''_%d = %.4e\n', i, Ap_fit);
    fprintf('  λ_0,%d = %.2f nm\n', i, lambda0_fit);
    fprintf('  σ_E,%d = %.4f eV\n', i, sigma_E_fit);
    fprintf('  峰位 = %.2f nm (k=0时固定)\n', lambda0_fit);
    
    % 计算该峰的积分强度
    I_peak = compute_single_peak(lambda_exp, Ap_fit, lambda0_fit, sigma_E_fit, C);
    peak_integral = trapz(lambda_exp, I_peak);
    fprintf('  积分强度 = %.4e\n', peak_integral);
end
fprintf('背景 B = %.4e\n', params_fit(end));

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
figure('Position', [100, 100, 1200, 700]);

% 主图：实验数据vs拟合
subplot(2,1,1);
plot(lambda_exp, I_exp, 'k.', 'MarkerSize', 4, 'DisplayName', '实验数据');
hold on;
plot(lambda_exp, I_fit, 'r-', 'LineWidth', 2, 'DisplayName', '总拟合');

% 绘制各个峰
colors = lines(n_peaks);
for i = 1:n_peaks
    idx = (i-1)*3 + 1;
    Ap = params_fit(idx);
    lambda0 = params_fit(idx+1);
    sigma_E = params_fit(idx+2);
    I_peak = compute_single_peak(lambda_exp, Ap, lambda0, sigma_E, C);
    plot(lambda_exp, I_peak, '--', 'Color', colors(i,:), ...
        'LineWidth', 1.5, 'DisplayName', sprintf('峰%d (%.1f nm)', i, lambda0));
end

% 绘制背景
B = params_fit(end);
plot(lambda_exp, ones(size(lambda_exp))*B, 'k:', 'LineWidth', 1, 'DisplayName', '背景');

xlabel('波长 (nm)');
ylabel('强度 (a.u.)');
legend('Location', 'best');
title(sprintf('量子点光谱拟合 (R^2 = %.4f)', R_squared));
grid on;
set(gca, 'FontSize', 11);

% 残差图
subplot(2,1,2);
plot(lambda_exp, residuals, 'b-', 'LineWidth', 1);
hold on;
plot(lambda_exp, zeros(size(lambda_exp)), 'k--');
xlabel('波长 (nm)');
ylabel('残差 (a.u.)');
title('拟合残差');
grid on;
set(gca, 'FontSize', 11);

%% 辅助函数：计算多峰总谱 (简化模型, k_m = 0)
function S = model_spectrum_simple(lambda, params, n_peaks, C)
    S = zeros(size(lambda));
    B = params(end);
    for i = 1:n_peaks
        idx = (i-1)*3 + 1;
        Ap = params(idx);
        lambda0 = params(idx+1);
        sigma_E = params(idx+2);
        
        % 使用简化公式: E_0 = C/lambda0 (当 k=0 时)
        % 但根据您的公式，还有第二项修正
        E0 = C / lambda0 - 2*sigma_E^2/C * lambda0;
        
        % 单峰谱形 I_m(lambda)
        Im = (Ap ./ lambda.^2) .* exp(-((C./lambda - E0).^2) ./ (2*sigma_E^2));
        S = S + Im;
    end
    S = S + B;
end

%% 辅助函数：计算单个峰
function I = compute_single_peak(lambda, Ap, lambda0, sigma_E, C)
    E0 = C / lambda0 - 2*sigma_E^2/C * lambda0;
    I = (Ap ./ lambda.^2) .* exp(-((C./lambda - E0).^2) ./ (2*sigma_E^2));
end

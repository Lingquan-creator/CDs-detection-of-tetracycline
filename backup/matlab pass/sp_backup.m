% sp.m - 量子点光谱拟合（严格模型，峰位-强度耦合）
%
% 根据公式：
% L_m(A'_m) = k_m * A'_m + λ_{0,m}  (峰位-强度关系)
% E_{0,m}(A'_m) = C/(k_m*A'_m + λ_{0,m}) - 2σ²_{E,m}/C * (k_m*A'_m + λ_{0,m})
% I_m(λ; A'_m) = (A'_m/λ²) * exp[-(C/λ - E_{0,m})² / (2σ²_{E,m})] + B
%

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
n_peaks = 3; % 峰数量（根据实际调整）

%% 3. 定义单峰谱形函数
% 中心能量 E_{0,m}(A'_m)
E0_func = @(Ap, km, lambda0m, sigma_Em) ...
    C./(km*Ap + lambda0m) - 2*sigma_Em.^2/C*(km*Ap + lambda0m);

% 单峰谱形 I_m(lambda; A'_m) （不包含背景）
Im_func = @(lambda, Ap, km, lambda0m, sigma_Em) ...
    (Ap./lambda.^2) .* exp(-((C./lambda - E0_func(Ap, km, lambda0m, sigma_Em)).^2) ...
    ./ (2*sigma_Em.^2));

%% 4. 定义多峰总谱函数
model_func = @(params, lambda) sum_peaks(lambda, params, n_peaks, C);

%% 5. 初始参数设置
% params = [A'_1, k_1, lambda0_1, sigma_E1, ..., A'_n, k_n, lambda0_n, sigma_En, B]
% 估计数据的峰位置
[pks, locs] = findpeaks(I_exp, lambda_exp, 'NPeaks', n_peaks, 'SortStr', 'descend');
if length(locs) < n_peaks
    % 如果找不到足够的峰，使用均匀分布
    locs = linspace(min(lambda_exp), max(lambda_exp), n_peaks+2);
    locs = locs(2:end-1);
end

% 估计背景
B0 = min(I_exp);

params0 = [];
for i = 1:n_peaks
    % A'_m: 峰强度初值
    % 对于 k_m = 0 的情况，在峰位处: I(lambda0) ≈ A' / lambda0^2
    % 因此 A' ≈ I_peak * lambda0^2
    if i <= length(pks)
        A0 = (pks(i) - B0) * locs(i)^2;  % 正确缩放
    else
        A0 = (max(I_exp) - B0) * locs(i)^2 / n_peaks;
    end
    % k_m: 峰位-强度斜率，设为0（假设峰位不随强度变化）
    % 如果实验中有峰位-强度关系数据，应该从那里拟合得到k_m
    k0 = 0;
    % lambda0_m: 峰位初值，从数据估计
    lambda0 = locs(i);
    % sigma_E,m: 能量展宽，根据典型波长计算能量宽度
    % 对于可见光量子点，典型半高全宽约30-50nm，对应能量约0.15 eV
    sigma_E0 = 0.12;
    params0 = [params0, A0, k0, lambda0, sigma_E0]; 
end
params0 = [params0, B0]; % 背景B

fprintf('\n初始参数:\n');
for i = 1:n_peaks
    idx = (i-1)*4 + 1;
    fprintf('峰 %d: A''=%.2e, k=%.4f, λ0=%.2f nm, σE=%.4f eV\n', ...
        i, params0(idx), params0(idx+1), params0(idx+2), params0(idx+3));
end
fprintf('背景 B = %.2e\n', params0(end));

%% 6. 参数约束
% 对每个峰: [A'_m (强度), k_m (斜率), lambda0_m (峰位), sigma_E,m (能量展宽)]
lb = [];
ub = [];
for i = 1:n_peaks
    lb = [lb, 0, -0.5, min(lambda_exp), 0.01];  % A'>=0, k>=-0.5, lambda0>=min, sigma_E>=0.01
    ub = [ub, Inf, 0.5, max(lambda_exp), 0.3];   % A'无上限, k<=0.5, lambda0<=max, sigma_E<=0.3
end
lb = [lb, 0];                % 背景 B >= 0
ub = [ub, max(I_exp)*0.5];   % 背景 B <= 最大强度的50%

%% 7. 非线性拟合
options = optimoptions('lsqcurvefit', ...
    'Display', 'iter', ...
    'MaxIterations', 2000, ...
    'MaxFunctionEvaluations', 10000, ...
    'FunctionTolerance', 1e-8, ...
    'StepTolerance', 1e-8);
params_fit = lsqcurvefit(model_func, params0, lambda_exp, I_exp, lb, ub, options);

%% 8. 计算拟合结果
I_fit = model_func(params_fit, lambda_exp);

%% 9. 绘图
figure('Position', [100, 100, 1200, 700]);

% 主图：实验数据vs拟合
subplot(2,1,1);
plot(lambda_exp, I_exp, 'k.', 'MarkerSize', 4, 'DisplayName', '实验数据');
hold on;
plot(lambda_exp, I_fit, 'r-', 'LineWidth', 2, 'DisplayName', '总拟合');

% 绘制各个峰
colors = lines(n_peaks);
for i = 1:n_peaks
    idx = (i-1)*4 + 1;
    Ap = params_fit(idx);
    km = params_fit(idx+1);
    lambda0m = params_fit(idx+2);
    sigma_Em = params_fit(idx+3);
    I_peak = Im_func(lambda_exp, Ap, km, lambda0m, sigma_Em);
    plot(lambda_exp, I_peak, '--', 'Color', colors(i,:), ...
        'LineWidth', 1.5, 'DisplayName', sprintf('峰%d (%.1f nm)', i, lambda0m));
end

% 绘制背景
B = params_fit(end);
plot(lambda_exp, ones(size(lambda_exp))*B, 'k:', 'LineWidth', 1, 'DisplayName', '背景');

xlabel('波长 (nm)');
ylabel('强度 (a.u.)');
legend('Location', 'best');
title('量子点光谱拟合（峰位-强度耦合模型）');
grid on;
set(gca, 'FontSize', 11);

% 残差图
subplot(2,1,2);
residuals = I_exp - I_fit;
plot(lambda_exp, residuals, 'b-', 'LineWidth', 1);
hold on;
plot(lambda_exp, zeros(size(lambda_exp)), 'k--');
xlabel('波长 (nm)');
ylabel('残差 (a.u.)');
title('拟合残差');
grid on;
set(gca, 'FontSize', 11);

%% 10. 输出拟合参数
fprintf('\n===== 拟合参数 =====\n');
for i = 1:n_peaks
    idx = (i-1)*4 + 1;
    Ap_fit = params_fit(idx);
    km_fit = params_fit(idx+1);
    lambda0m_fit = params_fit(idx+2);
    sigma_Em_fit = params_fit(idx+3);
    % 根据公式计算峰位: L_m(A'_m) = k_m * A'_m + lambda_0,m
    peak_pos = km_fit * Ap_fit + lambda0m_fit;
    fprintf('峰 %d:\n', i);
    fprintf('  A''_%d = %.4e\n', i, Ap_fit);
    fprintf('  k_%d = %.6f\n', i, km_fit);
    fprintf('  λ_0,%d = %.2f nm\n', i, lambda0m_fit);
    fprintf('  σ_E,%d = %.4f eV\n', i, sigma_Em_fit);
    if abs(km_fit) < 1e-6
        fprintf('  实际峰位 L_%d(A''_%d) ≈ %.2f nm (k≈0, 峰位固定)\n', i, i, lambda0m_fit);
    else
        fprintf('  实际峰位 L_%d(A''_%d) = %.2f nm\n', i, i, peak_pos);
    end
    % 计算该峰的积分强度
    I_peak_m = Im_func(lambda_exp, Ap_fit, km_fit, lambda0m_fit, sigma_Em_fit);
    peak_integral = trapz(lambda_exp, I_peak_m);
    fprintf('  积分强度 = %.4e\n', peak_integral);
end
fprintf('背景 B = %.4e\n', params_fit(end));

%% 11. 计算拟合质量
residuals = I_exp - I_fit;
SS_res = sum(residuals.^2);
SS_tot = sum((I_exp - mean(I_exp)).^2);
R_squared = 1 - SS_res / SS_tot;
RMSE = sqrt(mean(residuals.^2));
fprintf('\n===== 拟合质量 =====\n');
fprintf('R² = %.6f\n', R_squared);
fprintf('RMSE = %.4e\n', RMSE);
fprintf('相对RMSE = %.2f%%\n', 100 * RMSE / mean(I_exp));

%% 辅助函数：计算多峰总谱
function S = sum_peaks(lambda, params, n_peaks, C)
    S = zeros(size(lambda));
    B = params(end);
    for i = 1:n_peaks
        idx = (i-1)*4 + 1;
        Ap = params(idx);
        km = params(idx+1);
        lambda0m = params(idx+2);
        sigma_Em = params(idx+3);
        
        % 计算峰位波长 (考虑数值稳定性)
        if abs(km) < 1e-10
            % 当k接近0时，使用lambda0m作为峰位
            lambda_peak = lambda0m;
        else
            lambda_peak = km * Ap + lambda0m;
        end
        
        % 计算中心能量 E_0
        E0 = C / lambda_peak - 2*sigma_Em^2/C * lambda_peak;
        
        % 单峰谱形
        Im = (Ap./lambda.^2) .* exp(-((C./lambda - E0).^2) ./ (2*sigma_Em.^2));
        S = S + Im;
    end
    S = S + B;
end
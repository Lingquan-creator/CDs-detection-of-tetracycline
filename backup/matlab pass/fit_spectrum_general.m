% fit_spectrum_general.m - 通用量子点光谱拟合程序
%
% 严格物理模型：
% 峰位-强度关系: L_m(A'_m) = k_m * A'_m + λ_{0,m}
% 中心能量: E_{0,m}(A'_m) = C/(k_m*A'_m + λ_{0,m}) - 2σ²_{E,m}/C * (k_m*A'_m + λ_{0,m})
% 单峰谱形: I_m(λ; A'_m) = (A'_m/λ²) * exp[-(C/λ - E_{0,m})² / (2σ²_{E,m})]
% 多峰总谱: S(λ) = Σ I_m(λ; A'_m) + B
%
% 作者: [Your Name]
% 日期: 2025-12-12

clear; clc; close all;

%% ========================================================================
%  ===================== 用户自定义参数区域 ==============================
%  ========================================================================

% --- 数据文件设置 ---
data_file = 'yu 3.csv';          % 数据文件名
skip_rows = 21;                   % 跳过的行数（头部信息）
x_column = 1;                     % X坐标列（波长）
y_column = 4;                     % Y坐标列（强度）- 可修改为3, 4, 5等

% --- 峰数量设置 ---
n_peaks = 3;                      % 拟合的峰数量

% --- 峰参数设置 ---
% 为每个峰指定参数，使用NaN表示需要拟合确定
%
% 重要提示：建议固定 k_m = 0 (峰位不随强度变化)
% 如需拟合k_m，需要从独立的峰位-强度关系实验中确定
%
% 格式: [k_m, lambda0_m, sigma_E_m]
% k_m: 峰位-强度斜率（建议设为0）
% lambda0_m: 初始峰位 (nm，NaN表示自动检测)
% sigma_E_m: 能量展宽 (eV，NaN表示自动拟合)
%
% 使用说明:
% - 用 0 固定k_m=0（推荐，峰位不随强度变化）
% - 用 NaN 让参数自由拟合
% - 用具体数值固定该参数为指定值
%
% 示例1：全部自动拟合（k=0）
% peak_params = {};  % 空数组，k默认为0，其他自动
%
% 示例2：明确指定固定k=0，其他自动
% peak_params = {[0, NaN, NaN]; [0, NaN, NaN]; [0, NaN, NaN]};
%
% 示例3：指定某峰位置，固定
% peak_params = {[0, 450, NaN]; [0, NaN, NaN]};

peak_params = {};  % 默认：k=0，其他参数自动检测和拟合

% 注意：如果peak_params为空或行数 < n_peaks，剩余峰将自动检测参数

% --- 背景参数 ---
B_initial = NaN;                  % 背景初值，NaN表示自动估计为min(I_exp)

% --- 参数约束 ---
A_min = 0;                        % A'的最小值
A_max = Inf;                      % A'的最大值
k_min = -0.5;                     % k_m的最小值
k_max = 0.5;                      % k_m的最大值
sigma_E_min = 0.01;               % σ_E的最小值 (eV)
sigma_E_max = 0.5;                % σ_E的最大值 (eV)
B_min = 0;                        % 背景的最小值
B_max_ratio = 0.5;                % 背景最大值 = max(I_exp) * B_max_ratio

% --- 拟合设置 ---
max_iterations = 2000;            % 最大迭代次数
max_function_evals = 20000;       % 最大函数评估次数
display_iter = 'iter';            % 显示选项: 'off', 'iter', 'final'

% --- 输出设置 ---
save_results = false;              % 是否保存结果到文件
results_file = 'fit_results.txt'; % 结果文件名
save_figure = true;               % 是否保存图形
figure_file = 'fit_result.png';   % 图形文件名

%  ========================================================================
%  =================== 用户自定义参数区域结束 ============================
%  ========================================================================

%% 常数定义
C = 1239.8;  % eV·nm (光子能量-波长转换常数)

%% 1. 读取数据
fprintf('读取数据文件: %s\n', data_file);
data = readmatrix(data_file);

% 跳过头部行
if skip_rows > 0
    data = data(skip_rows+1:end, :);
end

% 提取X和Y数据
lambda_exp = data(:, x_column);
I_exp = data(:, y_column);

% 移除NaN和Inf值
valid_idx = isfinite(lambda_exp) & isfinite(I_exp);
lambda_exp = lambda_exp(valid_idx);
I_exp = I_exp(valid_idx);

fprintf('有效数据点数: %d\n', length(lambda_exp));
fprintf('波长范围: %.1f ~ %.1f nm\n', min(lambda_exp), max(lambda_exp));
fprintf('强度范围: %.2e ~ %.2e\n', min(I_exp), max(I_exp));

%% 2. 初始化峰参数
fprintf('\n=== 峰参数初始化 ===\n');

% 自动检测峰位置
[pks, locs] = findpeaks(I_exp, lambda_exp, 'NPeaks', n_peaks, 'SortStr', 'descend');
if length(locs) < n_peaks
    locs = linspace(min(lambda_exp)+50, max(lambda_exp)-50, n_peaks);
    pks = repmat(max(I_exp), n_peaks, 1);
end
locs = sort(locs);
[~, sort_idx] = sort(locs);
pks = pks(sort_idx);

% 估计背景
if isnan(B_initial)
    B0 = min(I_exp);
else
    B0 = B_initial;
end

% 构建初始参数向量和约束
params0 = [];
lb = [];
ub = [];
param_info = struct('peak', {}, 'k_fixed', {}, 'lambda0_fixed', {}, 'sigma_E_fixed', {});

for i = 1:n_peaks
    % 获取当前峰的参数设置
    if i <= length(peak_params)
        k_m = peak_params{i}(1);
        lambda0_m = peak_params{i}(2);
        sigma_E_m = peak_params{i}(3);
    else
        k_m = 0;  % 默认k=0
        lambda0_m = NaN;
        sigma_E_m = NaN;
    end
    
    % A'_m 初始值（总是需要拟合）
    if i <= length(pks)
        A0 = (pks(i) - B0) * locs(i)^2;
    else
        A0 = (max(I_exp) - B0) * locs(i)^2 / n_peaks;
    end
    A0 = max(A0, 1e6);  % 确保不为零或负数
    
    % k_m 参数
    if isnan(k_m)
        k0 = 0;  % 默认初值
        k_fixed = false;
    else
        k0 = k_m;
        k_fixed = true;
    end
    
    % lambda0_m 参数
    if isnan(lambda0_m)
        lambda0_init = locs(i);
        lambda0_fixed = false;
    else
        lambda0_init = lambda0_m;
        lambda0_fixed = true;
    end
    
    % sigma_E_m 参数
    if isnan(sigma_E_m)
        sigma_E_init = 0.12;  % 默认初值
        sigma_E_fixed = false;
    else
        sigma_E_init = sigma_E_m;
        sigma_E_fixed = true;
    end
    
    % 添加到参数向量
    params0 = [params0, A0, k0, lambda0_init, sigma_E_init];
    
    % 设置约束 - 考虑各参数是否固定
    % k的约束
    if k_fixed
        lb_k = k0;
        ub_k = k0;
    else
        lb_k = k_min;
        ub_k = k_max;
    end
    
    % lambda0的约束
    if lambda0_fixed
        lb_lambda0 = lambda0_init;
        ub_lambda0 = lambda0_init;
    else
        lb_lambda0 = min(lambda_exp);
        ub_lambda0 = max(lambda_exp);
    end
    
    % sigma_E的约束
    if sigma_E_fixed
        lb_sigma = sigma_E_init;
        ub_sigma = sigma_E_init;
    else
        lb_sigma = sigma_E_min;
        ub_sigma = sigma_E_max;
    end
    
    lb = [lb, A_min, lb_k, lb_lambda0, lb_sigma];
    ub = [ub, A_max, ub_k, ub_lambda0, ub_sigma];
    
    % 保存参数信息
    param_info(i).peak = i;
    param_info(i).k_fixed = k_fixed;
    param_info(i).lambda0_fixed = lambda0_fixed;
    param_info(i).sigma_E_fixed = sigma_E_fixed;
    
    fprintf('峰 %d 初始参数: A''=%.2e, k=%.4f%s, λ0=%.2f nm%s, σE=%.4f eV%s\n', ...
        i, A0, k0, tern(k_fixed, ' (固定)', ''), ...
        lambda0_init, tern(lambda0_fixed, ' (固定)', ''), ...
        sigma_E_init, tern(sigma_E_fixed, ' (固定)', ''));
end

% 添加背景参数
params0 = [params0, B0];
lb = [lb, B_min];
ub = [ub, max(I_exp) * B_max_ratio];

fprintf('背景 B 初始值: %.4e\n', B0);

%% 3. 定义模型函数
model_func = @(params, lambda) multi_peak_model(lambda, params, n_peaks, C);

%% 4. 非线性拟合
fprintf('\n=== 开始非线性拟合 ===\n');
options = optimoptions('lsqcurvefit', ...
    'Display', display_iter, ...
    'MaxIterations', max_iterations, ...
    'MaxFunctionEvaluations', max_function_evals, ...
    'FunctionTolerance', 1e-8, ...
    'StepTolerance', 1e-8);

params_fit = lsqcurvefit(model_func, params0, lambda_exp, I_exp, lb, ub, options);

%% 5. 计算拟合结果
I_fit = model_func(params_fit, lambda_exp);

% 计算拟合质量
residuals = I_exp - I_fit;
SS_res = sum(residuals.^2);
SS_tot = sum((I_exp - mean(I_exp)).^2);
R_squared = 1 - SS_res / SS_tot;
RMSE = sqrt(mean(residuals.^2));

%% 6. 输出拟合参数
fprintf('\n');
fprintf('========================================\n');
fprintf('           拟合结果\n');
fprintf('========================================\n\n');

for i = 1:n_peaks
    idx = (i-1)*4 + 1;
    Ap_fit = params_fit(idx);
    km_fit = params_fit(idx+1);
    lambda0_fit = params_fit(idx+2);
    sigma_E_fit = params_fit(idx+3);
    
    % 计算实际峰位
    if abs(km_fit) < 1e-10
        peak_pos = lambda0_fit;
    else
        peak_pos = km_fit * Ap_fit + lambda0_fit;
    end
    
    % 计算中心能量
    E0 = C / peak_pos - 2*sigma_E_fit^2/C * peak_pos;
    
    % 计算积分强度
    I_peak = compute_single_peak(lambda_exp, Ap_fit, km_fit, lambda0_fit, sigma_E_fit, C);
    peak_integral = trapz(lambda_exp, I_peak);
    
    fprintf('峰 %d:\n', i);
    fprintf('  A''_%d = %.6e\n', i, Ap_fit);
    fprintf('  k_%d = %.6f', i, km_fit);
    if param_info(i).k_fixed
        fprintf(' (固定)\n');
    else
        fprintf('\n');
    end
    fprintf('  λ_0,%d = %.4f nm', i, lambda0_fit);
    if param_info(i).lambda0_fixed
        fprintf(' (固定)\n');
    else
        fprintf('\n');
    end
    fprintf('  σ_E,%d = %.6f eV', i, sigma_E_fit);
    if param_info(i).sigma_E_fixed
        fprintf(' (固定)\n');
    else
        fprintf('\n');
    end
    fprintf('  实际峰位 = %.4f nm\n', peak_pos);
    fprintf('  中心能量 E_0 = %.6f eV\n', E0);
    fprintf('  积分强度 = %.6e\n', peak_integral);
    fprintf('\n');
end

B_fit = params_fit(end);
fprintf('背景 B = %.6e\n\n', B_fit);

fprintf('拟合质量:\n');
fprintf('  R² = %.8f\n', R_squared);
fprintf('  RMSE = %.6e\n', RMSE);
fprintf('  相对RMSE = %.4f%%\n\n', 100 * RMSE / mean(I_exp));

%% 7. 绘图
fig = figure('Position', [100, 100, 1200, 900]);

% 子图1: 实验数据 vs 拟合
subplot(3,1,1);
plot(lambda_exp, I_exp, 'k.', 'MarkerSize', 3, 'DisplayName', '实验数据');
hold on;
plot(lambda_exp, I_fit, 'r-', 'LineWidth', 2, 'DisplayName', '总拟合');

% 绘制各个峰
colors = lines(n_peaks);
for i = 1:n_peaks
    idx = (i-1)*4 + 1;
    Ap = params_fit(idx);
    km = params_fit(idx+1);
    lambda0 = params_fit(idx+2);
    sigma_E = params_fit(idx+3);
    I_peak = compute_single_peak(lambda_exp, Ap, km, lambda0, sigma_E, C);
    
    if abs(km) < 1e-10
        peak_pos = lambda0;
    else
        peak_pos = km * Ap + lambda0;
    end
    
    plot(lambda_exp, I_peak, '--', 'Color', colors(i,:), 'LineWidth', 1.5, ...
        'DisplayName', sprintf('峰%d (%.1f nm)', i, peak_pos));
end

% 绘制背景
plot(lambda_exp, ones(size(lambda_exp))*B_fit, 'k:', 'LineWidth', 1, ...
    'DisplayName', '背景');

xlabel('波长 (nm)', 'FontSize', 11);
ylabel('强度 (a.u.)', 'FontSize', 11);
legend('Location', 'best', 'FontSize', 9);
title(sprintf('多峰光谱拟合 (R^2 = %.6f, %d峰)', R_squared, n_peaks), 'FontSize', 12);
grid on;
set(gca, 'FontSize', 10);

% 子图2: 残差
subplot(3,1,2);
plot(lambda_exp, residuals, 'b-', 'LineWidth', 1);
hold on;
plot(lambda_exp, zeros(size(lambda_exp)), 'k--', 'LineWidth', 0.5);
xlabel('波长 (nm)', 'FontSize', 11);
ylabel('残差 (a.u.)', 'FontSize', 11);
title('拟合残差', 'FontSize', 12);
grid on;
set(gca, 'FontSize', 10);

% 子图3: 相对残差
subplot(3,1,3);
relative_residuals = 100 * residuals ./ I_exp;
plot(lambda_exp, relative_residuals, 'g-', 'LineWidth', 1);
hold on;
plot(lambda_exp, zeros(size(lambda_exp)), 'k--', 'LineWidth', 0.5);
xlabel('波长 (nm)', 'FontSize', 11);
ylabel('相对残差 (%)', 'FontSize', 11);
title('相对拟合残差', 'FontSize', 12);
grid on;
set(gca, 'FontSize', 10);

%% 8. 保存结果
if save_results
    fid = fopen(results_file, 'w');
    fprintf(fid, '========================================\n');
    fprintf(fid, '      量子点光谱拟合结果\n');
    fprintf(fid, '========================================\n\n');
    fprintf(fid, '数据文件: %s\n', data_file);
    fprintf(fid, '数据列: X=%d, Y=%d\n', x_column, y_column);
    fprintf(fid, '峰数量: %d\n\n', n_peaks);
    
    for i = 1:n_peaks
        idx = (i-1)*4 + 1;
        Ap_fit = params_fit(idx);
        km_fit = params_fit(idx+1);
        lambda0_fit = params_fit(idx+2);
        sigma_E_fit = params_fit(idx+3);
        
        if abs(km_fit) < 1e-10
            peak_pos = lambda0_fit;
        else
            peak_pos = km_fit * Ap_fit + lambda0_fit;
        end
        
        E0 = C / peak_pos - 2*sigma_E_fit^2/C * peak_pos;
        I_peak = compute_single_peak(lambda_exp, Ap_fit, km_fit, lambda0_fit, sigma_E_fit, C);
        peak_integral = trapz(lambda_exp, I_peak);
        
        fprintf(fid, '峰 %d:\n', i);
        fprintf(fid, '  A''_%d = %.8e\n', i, Ap_fit);
        fprintf(fid, '  k_%d = %.8f', i, km_fit);
        if param_info(i).k_fixed, fprintf(fid, ' (固定)'); end
        fprintf(fid, '\n');
        fprintf(fid, '  λ_0,%d = %.6f nm', i, lambda0_fit);
        if param_info(i).lambda0_fixed, fprintf(fid, ' (固定)'); end
        fprintf(fid, '\n');
        fprintf(fid, '  σ_E,%d = %.8f eV', i, sigma_E_fit);
        if param_info(i).sigma_E_fixed, fprintf(fid, ' (固定)'); end
        fprintf(fid, '\n');
        fprintf(fid, '  实际峰位 = %.6f nm\n', peak_pos);
        fprintf(fid, '  中心能量 E_0 = %.8f eV\n', E0);
        fprintf(fid, '  积分强度 = %.8e\n\n', peak_integral);
    end
    
    fprintf(fid, '背景 B = %.8e\n\n', B_fit);
    fprintf(fid, '拟合质量:\n');
    fprintf(fid, '  R² = %.10f\n', R_squared);
    fprintf(fid, '  RMSE = %.8e\n', RMSE);
    fprintf(fid, '  相对RMSE = %.6f%%\n', 100 * RMSE / mean(I_exp));
    fclose(fid);
    fprintf('结果已保存到: %s\n', results_file);
end

if save_figure
    saveas(fig, figure_file);
    fprintf('图形已保存到: %s\n', figure_file);
end

fprintf('\n拟合完成！\n');

%% ========================================================================
%  辅助函数
%  ========================================================================

%% 多峰模型函数
function S = multi_peak_model(lambda, params, n_peaks, C)
    S = zeros(size(lambda));
    B = params(end);
    
    for i = 1:n_peaks
        idx = (i-1)*4 + 1;
        Ap = params(idx);
        km = params(idx+1);
        lambda0 = params(idx+2);
        sigma_E = params(idx+3);
        
        % 计算峰位波长（数值稳定性）
        if abs(km) < 1e-10
            lambda_peak = lambda0;
        else
            lambda_peak = km * Ap + lambda0;
        end
        
        % 计算中心能量
        E0 = C / lambda_peak - 2*sigma_E^2/C * lambda_peak;
        
        % 单峰谱形
        Im = (Ap ./ lambda.^2) .* exp(-((C./lambda - E0).^2) ./ (2*sigma_E^2));
        S = S + Im;
    end
    
    S = S + B;
end

%% 计算单个峰
function I = compute_single_peak(lambda, Ap, km, lambda0, sigma_E, C)
    if abs(km) < 1e-10
        lambda_peak = lambda0;
    else
        lambda_peak = km * Ap + lambda0;
    end
    
    E0 = C / lambda_peak - 2*sigma_E^2/C * lambda_peak;
    I = (Ap ./ lambda.^2) .* exp(-((C./lambda - E0).^2) ./ (2*sigma_E^2));
end

%% 三元运算符辅助函数
function result = tern(condition, true_val, false_val)
    if condition
        result = true_val;
    else
        result = false_val;
    end
end

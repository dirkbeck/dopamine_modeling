clear; close all; clc;

filename = 'nHT.mat';

%% Load data
load(filename);

% Assign rats to groups
theoretical_CT = [1.5, 3, 4.5, 6, 9, 15, 20, 27, 36, 54, 72, 110, 180, 300];
rats = unique(nHT.Rat);
n_rats = length(rats);
rat_to_group = containers.Map('KeyType', 'double', 'ValueType', 'double');

for i = 1:n_rats
    rat_trials = nHT(nHT.Rat == rats(i), :);
    iota = mean(rat_trials.Inftns, 'omitnan');
    if ~isnan(iota) && iota > 0
        [~, group_idx] = min(abs(theoretical_CT - iota));
        rat_to_group(rats(i)) = group_idx;
    end
end

%% define acquisition detection methods
METHODS = struct();
method_idx = 1;

% Sustained threshold (3 trials)
METHODS(method_idx).name = 'CS>Context sustained 3';
METHODS(method_idx).short_name = 'CS>Ctx sus3';
METHODS(method_idx).threshold_label = 'CS > Context';
METHODS(method_idx).detection_label = 'Sustained 3';
METHODS(method_idx).func = @(rat_trials) detect_sustained_threshold(rat_trials.signed_nDkl, 0, 3);
method_idx = method_idx + 1;

METHODS(method_idx).name = 'Odds 4:1 sustained 3';
METHODS(method_idx).short_name = 'Odds4to1 sus3';
METHODS(method_idx).threshold_label = 'Odds 4:1';
METHODS(method_idx).detection_label = 'Sustained 3';
METHODS(method_idx).func = @(rat_trials) detect_sustained_threshold(rat_trials.signed_nDkl, 0.82, 3);
method_idx = method_idx + 1;

METHODS(method_idx).name = 'p<0.05 sustained 3';
METHODS(method_idx).short_name = 'p<0.05 sus3';
METHODS(method_idx).threshold_label = 'p < 0.05';
METHODS(method_idx).detection_label = 'Sustained 3';
METHODS(method_idx).func = @(rat_trials) detect_sustained_threshold(rat_trials.signed_nDkl, 1.92, 3);
method_idx = method_idx + 1;

% permanent threshold
METHODS(method_idx).name = 'CS>Context permanent';
METHODS(method_idx).short_name = 'CS>Ctx perm';
METHODS(method_idx).threshold_label = 'CS > Context';
METHODS(method_idx).detection_label = 'Permanent';
METHODS(method_idx).func = @(rat_trials) detect_permanent_threshold(rat_trials.signed_nDkl, 0);
method_idx = method_idx + 1;

METHODS(method_idx).name = 'Odds 4:1 permanent';
METHODS(method_idx).short_name = 'Odds4to1 perm';
METHODS(method_idx).threshold_label = 'Odds 4:1';
METHODS(method_idx).detection_label = 'Permanent';
METHODS(method_idx).func = @(rat_trials) detect_permanent_threshold(rat_trials.signed_nDkl, 0.82);
method_idx = method_idx + 1;

METHODS(method_idx).name = 'p<0.05 permanent';
METHODS(method_idx).short_name = 'p<0.05 perm';
METHODS(method_idx).threshold_label = 'p < 0.05';
METHODS(method_idx).detection_label = 'Permanent';
METHODS(method_idx).func = @(rat_trials) detect_permanent_threshold(rat_trials.signed_nDkl, 1.92);
method_idx = method_idx + 1;

% First crossing
METHODS(method_idx).name = 'Odds 4:1 first cross';
METHODS(method_idx).short_name = 'Odds4to1 1st';
METHODS(method_idx).threshold_label = 'Odds 4:1';
METHODS(method_idx).detection_label = 'First crossing';
METHODS(method_idx).func = @(rat_trials) detect_first_crossing(rat_trials.signed_nDkl, 0.82);
method_idx = method_idx + 1;

METHODS(method_idx).name = 'p<0.05 first cross';
METHODS(method_idx).short_name = 'p<0.05 1st';
METHODS(method_idx).threshold_label = 'p < 0.05';
METHODS(method_idx).detection_label = 'First crossing';
METHODS(method_idx).func = @(rat_trials) detect_first_crossing(rat_trials.signed_nDkl, 1.92);
method_idx = method_idx + 1;

% Cumulative nDKL
METHODS(method_idx).name = 'Odds 4:1 cumulative';
METHODS(method_idx).short_name = 'Odds4to1 cum';
METHODS(method_idx).threshold_label = 'Odds 4:1';
METHODS(method_idx).detection_label = 'Cumulative';
METHODS(method_idx).func = @(rat_trials) detect_cumulative_ndkl(rat_trials, 0.82);
method_idx = method_idx + 1;

METHODS(method_idx).name = 'p<0.05 cumulative';
METHODS(method_idx).short_name = 'p<0.05 cum';
METHODS(method_idx).threshold_label = 'p < 0.05';
METHODS(method_idx).detection_label = 'Cumulative';
METHODS(method_idx).func = @(rat_trials) detect_cumulative_ndkl(rat_trials, 1.92);
method_idx = method_idx + 1;

n_methods = length(METHODS);

%% Test all methods - first 10 groups
all_results = [];

for m = 1:n_methods
    method = METHODS(m);
    
    rat_data = [];
    for i = 1:n_rats
        if ~isKey(rat_to_group, rats(i))
            continue;
        end
        
        group_id = rat_to_group(rats(i));
        if group_id > 10
            continue;
        end
        
        rat_trials = nHT(nHT.Rat == rats(i), :);
        iota = mean(rat_trials.Inftns, 'omitnan');
        
        if isnan(iota) || iota <= 0
            continue;
        end
        
        try
            [acq_trial, acquired] = method.func(rat_trials);
            
            if acquired && acq_trial > 0 && acq_trial <= height(rat_trials)
                rat_data(end+1).rat = rats(i);
                rat_data(end).group = group_id;
                rat_data(end).informativeness = iota;
                rat_data(end).trials_to_acq = acq_trial;
            end
        catch
            continue;
        end
    end
    
    if isempty(rat_data)
        continue;
    end
    
    data = struct2table(rat_data);
    
    %% FIT MODELS ON GROUP MEDIANS
    group_data = [];
    for g = 1:10
        group_rats = data(data.group == g, :);
        if height(group_rats) > 0
            group_data(end+1).group_id = g;
            group_data(end).actual_CT = mean(group_rats.informativeness);
            group_data(end).median_trials = median(group_rats.trials_to_acq);
            group_data(end).n_rats = height(group_rats);
        end
    end
    group_data = struct2table(group_data);
    
    if height(group_data) < 5
        continue;
    end
    
    % informativenss on group medians
    log_iota = log(group_data.actual_CT - 1);
    log_trials = log(group_data.median_trials);
    valid = isfinite(log_iota) & isfinite(log_trials);
    
    if sum(valid) < 3
        continue;
    end
    
    p_gal = polyfit(log_iota(valid), log_trials(valid), 1);
    slope_gal = p_gal(1);
    k_gal = exp(p_gal(2));
    
    pred_log_gal = polyval(p_gal, log_iota(valid));
    ss_tot_log = sum((log_trials(valid) - mean(log_trials(valid))).^2);
    ss_res_log = sum((log_trials(valid) - pred_log_gal).^2);
    r2_gal_group = 1 - ss_res_log / ss_tot_log;
    
    % policy-IG on group medians
    X_pig = [ones(sum(valid), 1), 1./log(group_data.actual_CT(valid))];
    beta_pig = X_pig \ group_data.median_trials(valid);
    pred_pig = X_pig * beta_pig;
    
    ss_tot_lin = sum((group_data.median_trials(valid) - mean(group_data.median_trials(valid))).^2);
    ss_res_pig = sum((group_data.median_trials(valid) - pred_pig).^2);
    r2_pig_group = 1 - ss_res_pig / ss_tot_lin;
    
    % TD-RPE on group medians
    r2_td_group = 0;
    mean_trials = mean(group_data.median_trials(valid));
    
    %% fit models on individual rats
    log_iota_ind = log(data.informativeness - 1);
    log_trials_ind = log(data.trials_to_acq);
    valid_ind = isfinite(log_iota_ind) & isfinite(log_trials_ind);
    
    if sum(valid_ind) < 3
        r2_gal_ind = NaN;
        r2_pig_ind = NaN;
        r2_td_ind = NaN;
    else
        p_gal_ind = polyfit(log_iota_ind(valid_ind), log_trials_ind(valid_ind), 1);
        pred_log_gal_ind = polyval(p_gal_ind, log_iota_ind(valid_ind));
        
        ss_tot_log_ind = sum((log_trials_ind(valid_ind) - mean(log_trials_ind(valid_ind))).^2);
        ss_res_log_ind = sum((log_trials_ind(valid_ind) - pred_log_gal_ind).^2);
        r2_gal_ind = 1 - ss_res_log_ind / ss_tot_log_ind;
        
        % Policy-IG on individual rats
        X_pig_ind = [ones(sum(valid_ind), 1), 1./log(data.informativeness(valid_ind))];
        beta_pig_ind = X_pig_ind \ data.trials_to_acq(valid_ind);
        pred_pig_ind = X_pig_ind * beta_pig_ind;
        
        ss_tot_lin_ind = sum((data.trials_to_acq(valid_ind) - mean(data.trials_to_acq(valid_ind))).^2);
        ss_res_pig_ind = sum((data.trials_to_acq(valid_ind) - pred_pig_ind).^2);
        r2_pig_ind = 1 - ss_res_pig_ind / ss_tot_lin_ind;
        
        % TD-RPE on individual rats
        r2_td_ind = 0;
    end
    
    all_results(end+1).method = method.name;
    all_results(end).short_name = method.short_name;
    all_results(end).threshold_label = method.threshold_label;
    all_results(end).detection_label = method.detection_label;
    all_results(end).n_rats = height(data);
    all_results(end).n_groups = sum(valid);
    all_results(end).slope = slope_gal;
    all_results(end).k_gal = k_gal;
    
    % Group-level R2
    all_results(end).r2_gal_group = r2_gal_group;
    all_results(end).r2_pig_group = r2_pig_group;
    all_results(end).r2_td_group = r2_td_group;
    
    % Session-level R2
    all_results(end).r2_gal_session = r2_gal_ind;
    all_results(end).r2_pig_session = r2_pig_ind;
    all_results(end).r2_td_session = r2_td_ind;
    
    all_results(end).group_data = group_data;
    all_results(end).data = data;
    all_results(end).beta_pig = beta_pig;
    all_results(end).p_gal = p_gal;
    all_results(end).mean_trials = mean_trials;
end

%% Select best method
best_idx = find(contains({all_results.method}, 'p<0.05') & contains({all_results.method}, 'sustained'));
if isempty(best_idx)
    best_idx = 1;
end
best_result = all_results(best_idx);

fprintf('GROUP-LEVEL: Informativeness R2=%.3f, Policy-IG R2=%.3f, TD-RPE R2=%.3f\n', ...
    best_result.r2_gal_group, best_result.r2_pig_group, best_result.r2_td_group);
fprintf('SESSION-LEVEL: Informativeness R2=%.3f, Policy-IG R2=%.3f, TD-RPE R2=%.3f\n\n', ...
    best_result.r2_gal_session, best_result.r2_pig_session, best_result.r2_td_session);

%% prepare data for plotting
gd = best_result.group_data;
d = best_result.data;

log_iota_g = log(gd.actual_CT - 1);
log_trials_g = log(gd.median_trials);
valid_g = isfinite(log_iota_g) & isfinite(log_trials_g);

% Model predictions
pred_log_gal = polyval(best_result.p_gal, log_iota_g(valid_g));

policy_IG_g = log(gd.actual_CT(valid_g));
X_pig_g = [ones(size(policy_IG_g)), 1./policy_IG_g];
pred_trials_pig = X_pig_g * best_result.beta_pig;

mean_trials = best_result.mean_trials;

%% plots
fig_w = 6.5;
fig_h = 3.2;
fig = figure('Units', 'inches', 'Position', [0.5 0.5 fig_w fig_h], 'Color', 'w', ...
    'PaperUnits', 'inches', 'PaperSize', [fig_w fig_h], ...
    'PaperPosition', [0 0 fig_w fig_h]);
set(fig, 'Renderer', 'painters');
set(fig, 'DefaultAxesFontName', 'Arial', 'DefaultTextFontName', 'Arial', ...
    'DefaultAxesFontSize', 6);

clean_ax = @(ax) set(ax, 'Box', 'off', 'Color', 'none', ...
    'TickDir', 'out', 'TickLength', [0.025 0.025], ...
    'FontSize', 6, 'FontName', 'Arial', 'LineWidth', 0.75);


n_panels = 5;
margin_l = 0.07; margin_r = 0.01;
margin_b = 0.32; margin_t = 0.08;
gap = 0.06;
panel_w = (1 - margin_l - margin_r - (n_panels-1)*gap) / n_panels;
panel_h = 1 - margin_b - margin_t;

for p = 1:n_panels
    ax_pos{p} = [margin_l + (p-1)*(panel_w + gap), margin_b, panel_w, panel_h];
end

% --- Panel 1: Gallistel Model ---
axes('Position', ax_pos{1});
hold on;

valid_rats = d.informativeness > 1;
scatter(log(d.informativeness(valid_rats) - 1), log(d.trials_to_acq(valid_rats)), ...
    15, [0.7 0.7 0.7], 'filled', 'MarkerFaceAlpha', 0.3);

scatter(log_iota_g(valid_g), log_trials_g(valid_g), ...
    60, [0.2 0.4 0.8], 'filled', 'MarkerFaceAlpha', 0.8, ...
    'MarkerEdgeColor', 'k', 'LineWidth', 1);

x_range = [min(log_iota_g(valid_g)), max(log_iota_g(valid_g))];
y_theory = -1.0 * x_range + log(best_result.k_gal);
plot(x_range, y_theory, 'k--', 'LineWidth', 1.5);

y_actual = best_result.slope * x_range + log(best_result.k_gal);
plot(x_range, y_actual, 'b-', 'LineWidth', 2);

xlabel('ln(informativeness - 1)', 'FontSize', 6, 'FontName', 'Arial');
ylabel('ln(trials to acquisition)', 'FontSize', 6, 'FontName', 'Arial');
title(sprintf('Informativeness model\n(R^2 = %.3f)', best_result.r2_gal_group), ...
    'FontSize', 7, 'FontName', 'Arial');

ax = gca; clean_ax(ax);
xl = xlim; yl = ylim;
xt = [floor(xl(1)*10)/10, ceil(xl(2)*10)/10];
yt = [floor(yl(1)*10)/10, ceil(yl(2)*10)/10];
set(ax, 'XTick', xt, 'YTick', yt, 'XLim', xt, 'YLim', yt);

lg = legend({'Individual rats', 'Group medians', ...
    'Theory (slope = -1.0)', sprintf('Fit (slope = %.2f)', best_result.slope)}, ...
    'FontSize', 5, 'FontName', 'Arial', 'NumColumns', 2);
legend('boxoff');
lg.Units = 'normalized';
lg.Position(1) = ax_pos{1}(1);
lg.Position(2) = 0.02;

% --- Panel 2: Policy-IG Model ----
axes('Position', ax_pos{2});
hold on;

scatter(log(d.informativeness(valid_rats)), d.trials_to_acq(valid_rats), ...
    15, [0.7 0.7 0.7], 'filled', 'MarkerFaceAlpha', 0.3);

scatter(log(gd.actual_CT(valid_g)), gd.median_trials(valid_g), ...
    60, [0.9 0.3 0.3], 'filled', 'MarkerFaceAlpha', 0.8, ...
    'MarkerEdgeColor', 'k', 'LineWidth', 1);

x_range_lin = linspace(min(log(gd.actual_CT)), max(log(gd.actual_CT)), 100);
X_range = [ones(size(x_range_lin')), 1./x_range_lin'];
y_range = X_range * best_result.beta_pig;
plot(x_range_lin, y_range, 'r-', 'LineWidth', 2);

xlabel('ln(informativeness)', 'FontSize', 6, 'FontName', 'Arial');
ylabel('Trials to acquisition', 'FontSize', 6, 'FontName', 'Arial');
title(sprintf('Policy-IG model\n(R^2 = %.3f)', best_result.r2_pig_group), ...
    'FontSize', 7, 'FontName', 'Arial');

ax = gca; clean_ax(ax);
xl = xlim; yl = ylim;
xt = [floor(xl(1)*10)/10, ceil(xl(2)*10)/10];
yt = [floor(yl(1)), ceil(yl(2))];
set(ax, 'XTick', xt, 'YTick', yt, 'XLim', xt, 'YLim', yt);

lg = legend({'Individual rats', 'Group medians', 'Policy-IG fit'}, ...
    'FontSize', 5, 'FontName', 'Arial', 'NumColumns', 2);
legend('boxoff');
lg.Units = 'normalized';
lg.Position(1) = ax_pos{2}(1);
lg.Position(2) = 0.02;

% --- Panel 3: TD-RPE Model -----
axes('Position', ax_pos{3});
hold on;

scatter(log(d.informativeness(valid_rats)), d.trials_to_acq(valid_rats), ...
    15, [0.7 0.7 0.7], 'filled', 'MarkerFaceAlpha', 0.3);

scatter(log(gd.actual_CT(valid_g)), gd.median_trials(valid_g), ...
    60, [0.3 0.7 0.3], 'filled', 'MarkerFaceAlpha', 0.8, ...
    'MarkerEdgeColor', 'k', 'LineWidth', 1);

plot([min(log(gd.actual_CT)), max(log(gd.actual_CT))], ...
    [mean_trials, mean_trials], 'g-', 'LineWidth', 2);

xlabel('ln(informativeness)', 'FontSize', 6, 'FontName', 'Arial');
ylabel('Trials to acquisition', 'FontSize', 6, 'FontName', 'Arial');
title(sprintf('TD-RPE null model\n(R^2 = %.3f)', best_result.r2_td_group), ...
    'FontSize', 7, 'FontName', 'Arial');

ax = gca; clean_ax(ax);
xl = xlim; yl = ylim;
xt = [floor(xl(1)*10)/10, ceil(xl(2)*10)/10];
yt = [floor(yl(1)), ceil(yl(2))];
set(ax, 'XTick', xt, 'YTick', yt, 'XLim', xt, 'YLim', yt);

lg = legend({'Individual rats', 'Group medians', 'Mean (no effect)'}, ...
    'FontSize', 5, 'FontName', 'Arial', 'NumColumns', 2);
legend('boxoff');
lg.Units = 'normalized';
lg.Position(1) = ax_pos{3}(1);
lg.Position(2) = 0.02;


n_methods_plot = length(all_results);
barplot_labels = cell(n_methods_plot, 1);
for i = 1:n_methods_plot
    barplot_labels{i} = sprintf('%s\n%s', ...
        all_results(i).threshold_label, all_results(i).detection_label);
end

% --- Panel 4: GROUP-LEVEL R2 bar plot ---
axes('Position', ax_pos{4});
hold on;

r2_matrix_group = zeros(n_methods_plot, 3);
for i = 1:n_methods_plot
    r2_matrix_group(i, 1) = all_results(i).r2_gal_group;
    r2_matrix_group(i, 2) = all_results(i).r2_pig_group;
    r2_matrix_group(i, 3) = all_results(i).r2_td_group;
end

x = 1:n_methods_plot;
width = 0.25;

bar(x - width, r2_matrix_group(:,1), width, 'FaceColor', [0.2 0.4 0.8], ...
    'EdgeColor', 'k', 'LineWidth', 0.5);
bar(x, r2_matrix_group(:,2), width, 'FaceColor', [0.9 0.3 0.3], ...
    'EdgeColor', 'k', 'LineWidth', 0.5);
bar(x + width, r2_matrix_group(:,3), width, 'FaceColor', [0.3 0.7 0.3], ...
    'EdgeColor', 'k', 'LineWidth', 0.5);

set(gca, 'XTick', x, 'XTickLabel', barplot_labels, 'XTickLabelRotation', 45);
ylabel('R^2', 'FontSize', 6, 'FontName', 'Arial');
title('Group-level (medians)', 'FontSize', 7, 'FontName', 'Arial');
ylim([0 1]);

ax = gca; clean_ax(ax);
set(ax, 'YTick', [0 1]);

lg = legend({'Informativeness', 'Policy-IG', 'TD-RPE'}, ...
    'FontSize', 5, 'FontName', 'Arial', 'NumColumns', 2);
legend('boxoff');
lg.Units = 'normalized';
lg.Position(1) = ax_pos{4}(1);
lg.Position(2) = 0.02;

% --- Panel 5: SESSION-LEVEL R2 bar plot ----
axes('Position', ax_pos{5});
hold on;

r2_matrix_session = zeros(n_methods_plot, 3);
for i = 1:n_methods_plot
    r2_matrix_session(i, 1) = all_results(i).r2_gal_session;
    r2_matrix_session(i, 2) = all_results(i).r2_pig_session;
    r2_matrix_session(i, 3) = all_results(i).r2_td_session;
end

bar(x - width, r2_matrix_session(:,1), width, 'FaceColor', [0.2 0.4 0.8], ...
    'EdgeColor', 'k', 'LineWidth', 0.5);
bar(x, r2_matrix_session(:,2), width, 'FaceColor', [0.9 0.3 0.3], ...
    'EdgeColor', 'k', 'LineWidth', 0.5);
bar(x + width, r2_matrix_session(:,3), width, 'FaceColor', [0.3 0.7 0.3], ...
    'EdgeColor', 'k', 'LineWidth', 0.5);

set(gca, 'XTick', x, 'XTickLabel', barplot_labels, 'XTickLabelRotation', 45);
ylabel('R^2', 'FontSize', 6, 'FontName', 'Arial');
title('Session-level (individual rats)', 'FontSize', 7, 'FontName', 'Arial');
ylim([0 1]);

ax = gca; clean_ax(ax);
set(ax, 'YTick', [0 1]);

lg = legend({'Informativeness', 'Policy-IG', 'TD-RPE'}, ...
    'FontSize', 5, 'FontName', 'Arial', 'NumColumns', 2);
legend('boxoff');
lg.Units = 'normalized';
lg.Position(1) = ax_pos{5}(1);
lg.Position(2) = 0.02;

figure(fig);
print(fig, 'model_comparison', '-depsc', '-painters');
saveas(fig, 'model_comparison.svg');

%% Helper functions

function [acq_trial, detected] = detect_permanent_threshold(signal, threshold)
    n = length(signal);
    detected = false;
    acq_trial = n;
    
    for t = 1:n
        if all(signal(t:end) >= threshold)
            acq_trial = t;
            detected = true;
            break;
        end
    end
end

function [acq_trial, detected] = detect_sustained_threshold(signal, threshold, n_sustained)
    n = length(signal);
    detected = false;
    acq_trial = n;
    
    for t = 1:(n - n_sustained + 1)
        if all(signal(t:min(t+n_sustained-1, n)) >= threshold)
            acq_trial = t;
            detected = true;
            break;
        end
    end
end

function [acq_trial, detected] = detect_first_crossing(signal, threshold)
    idx = find(signal >= threshold, 1, 'first');
    if ~isempty(idx)
        acq_trial = idx;
        detected = true;
    else
        acq_trial = length(signal);
        detected = false;
    end
end

function [acq_trial, detected] = detect_cumulative_ndkl(rat_trials, threshold)
    n = height(rat_trials);
    cumul_ndkl = zeros(n, 1);
    
    for t = 1:n
        cum_cs_rate = mean(rat_trials.r_CS(1:t), 'omitnan');
        cum_context_rate = mean(rat_trials.r_C(1:t), 'omitnan');
        
        if cum_cs_rate > 0 && cum_context_rate > 0 && cum_cs_rate ~= cum_context_rate
            cumul_ndkl(t) = t * abs(log(cum_cs_rate / cum_context_rate));
        end
    end
    
    [acq_trial, detected] = detect_permanent_threshold(cumul_ndkl, threshold);
end
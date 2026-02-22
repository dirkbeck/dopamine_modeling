clear; close all; clc;

dataDir = 'doi_10_5061_dryad_hhmgqnkjw__v20220602';
load(fullfile(dataDir, 'Amo_ExtractedPeaks.mat'));

cfg.min_points = 15;

col_pos  = [0.220 0.557 0.835];   % residual > 0
col_neg  = [0.839 0.153 0.157];   %  residual < 0
col_fit  = [0.15  0.15  0.15 ];   % exponential fit
col_data = [0.55  0.55  0.55 ];   % raw scattter

%% per-animal streaks analysis 
nA = length(FTL);
results = struct();
n_valid = 0;

for a = 1:nA
    pk = FTL(a).peaks;
    if isempty(pk), continue; end
    valid = ~isnan(pk(:,2));
    tr = pk(valid,1); pks = pk(valid,2);
    if length(tr) < cfg.min_points, continue; end

    % exponential fit
    ep = fitExpToData(tr, pks);
    pred = ep.Tinf + (ep.T0 - ep.Tinf) * exp(-ep.beta * tr);
    resid = pks - pred;

    signs = sign(resid);
    signs(signs == 0) = 1;

    % count streaks
    n_streaks = 1 + sum(diff(signs) ~= 0);
    n_pos  = sum(signs > 0);
    n_neg  = sum(signs < 0);
    nPts   = length(signs);

    % expected streaks and variance under H0 (iid)
    E_streaks = 1 + 2*n_pos*n_neg / nPts;
    V_streaks = 2*n_pos*n_neg*(2*n_pos*n_neg - nPts) / (nPts^2 * (nPts - 1));
    z_streaks = (n_streaks - E_streaks) / sqrt(max(V_streaks, 1e-10));

    % streak-length distribution
    streak_starts = [1; find(diff(signs) ~= 0) + 1];
    streak_ends   = [find(diff(signs) ~= 0); nPts];
    streak_lengths = streak_ends - streak_starts + 1;
    streak_signs   = signs(streak_starts);

    n_valid = n_valid + 1;
    results(n_valid).id          = FTL(a).animalID;
    results(n_valid).tr          = tr;
    results(n_valid).pks         = pks;
    results(n_valid).ep          = ep;
    results(n_valid).pred        = pred;
    results(n_valid).resid       = resid;
    results(n_valid).signs       = signs;
    results(n_valid).n_streaks      = n_streaks;
    results(n_valid).E_streaks      = E_streaks;
    results(n_valid).z_streaks      = z_streaks;
    results(n_valid).p_streaks      = 2 * normcdf(z_streaks);
    results(n_valid).streak_lengths = streak_lengths;
    results(n_valid).streak_signs   = streak_signs;
    results(n_valid).nPts        = nPts;

    fprintf('  M%d: n=%d  streaks=%d  E[streaks]=%.0f  z=%.2f  p=%.2e\n', ...
        results(n_valid).id, nPts, n_streaks, E_streaks, z_streaks, results(n_valid).p_streaks);
end

%% Fisher combined test
p_vals = max([results.p_streaks], 1e-300);
chi2_fisher = -2 * sum(log(p_vals));
df_fisher   = 2 * n_valid;
p_fisher    = 1 - chi2cdf(chi2_fisher, df_fisher);
fprintf('\nFisher combined: chi2=%.1f  df=%d  p=%.2e\n', ...
    chi2_fisher, df_fisher, p_fisher);

%% plotting

fontName = 'Arial';

nCol_anim  = min(4, n_valid);
nRow_anim  = ceil(n_valid / nCol_anim) * 2;
nCol_total = max(nCol_anim, 4);
nRow_total = nRow_anim + 1;

fig = figure('Position', [30 30 nCol_total*340 nRow_total*155], ...
    'Color', 'w', 'Renderer', 'painters');

for vi = 1:n_valid
    r = results(vi);

    col_idx = mod(vi-1, nCol_anim) + 1;
    row_blk = floor((vi-1) / nCol_anim);
    sp_top  = row_blk * 2 * nCol_total + col_idx;
    sp_bot  = row_blk * 2 * nCol_total + nCol_total + col_idx;

    pos_mask = r.signs > 0;
    neg_mask = r.signs < 0;
    ax_top = subplot(nRow_total, nCol_total, sp_top);
    hold on;

    plot(r.tr, r.pks, '.', 'Color', [col_data 0.15], 'MarkerSize', 3);
    plot(r.tr(pos_mask), r.pks(pos_mask), '.', 'Color', [col_pos 0.55], 'MarkerSize', 5);
    plot(r.tr(neg_mask), r.pks(neg_mask), '.', 'Color', [col_neg 0.55], 'MarkerSize', 5);


    tSm = linspace(1, max(r.tr), 500);
    plot(tSm, r.ep.Tinf + (r.ep.T0 - r.ep.Tinf) * exp(-r.ep.beta * tSm), ...
        '-', 'Color', col_fit, 'LineWidth', 2);

    yline(3000, ':', 'Color', [.8 .8 .8]);
    ylabel('Latency (ms)', 'FontName', fontName, 'FontSize', 9);
    ylim([0 3400]);
    title(sprintf('M%d  (z = %.1f)', r.id, r.z_streaks), ...
        'FontName', fontName, 'FontSize', 12, 'FontWeight', 'bold');
    set(gca, 'Box', 'off', 'TickDir', 'out', 'FontName', fontName, ...
        'FontSize', 9, 'XTickLabel', []);

    if vi == 1
        legend({'', 'Above fit', 'Below fit', 'Exp fit'}, ...
            'Location', 'ne', 'FontSize', 8, 'Box', 'off');
    end

    ax_bot = subplot(nRow_total, nCol_total, sp_bot);
    hold on;

    streak_starts_tr = r.tr([1; find(diff(r.signs) ~= 0) + 1]);
    streak_ends_tr   = r.tr([find(diff(r.signs) ~= 0); r.nPts]);
    for ri = 1:length(r.streak_lengths)
        x0 = streak_starts_tr(ri);
        x1 = streak_ends_tr(ri);
        c_shade = tern(r.streak_signs(ri) > 0, col_pos, col_neg);
        fill([x0 x1 x1 x0], [-1 -1 1 1]*max(abs(r.resid))*1.05, ...
            c_shade, 'FaceAlpha', 0.10, 'EdgeColor', 'none');
    end

    plot(r.tr(pos_mask), r.resid(pos_mask), '.', 'Color', [col_pos 0.5], 'MarkerSize', 3);
    plot(r.tr(neg_mask), r.resid(neg_mask), '.', 'Color', [col_neg 0.5], 'MarkerSize', 3);
    yline(0, '-', 'Color', [.4 .4 .4], 'LineWidth', 0.8);

    ylabel('Resid (ms)', 'FontName', fontName, 'FontSize', 9);
    xlabel('Trial', 'FontName', fontName, 'FontSize', 9);
    yl = max(abs(r.resid)) * 1.1;
    ylim([-yl yl]);
    set(gca, 'Box', 'off', 'TickDir', 'out', 'FontName', fontName, 'FontSize', 9);

    linkaxes([ax_top, ax_bot], 'x');
    xlim(ax_top, [0 max(r.tr)+10]);
end

sp_sumA = sub2ind_span(nRow_total, nCol_total, nRow_total, 1:floor(nCol_total/2));
ax_zA = subplot(nRow_total, nCol_total, sp_sumA);
hold on;

ids    = [results.id];
zvals  = [results.z_streaks];
[~, ord] = sort(zvals);

for i = 1:n_valid
    ai = ord(i);
    plot([0 zvals(ai)], [i i], '-', 'Color', [.6 .6 .6], 'LineWidth', 1.4);
    plot(zvals(ai), i, 'o', 'MarkerSize', 6, ...
        'MarkerFaceColor', col_neg, 'MarkerEdgeColor', 'none');
end

xline(-1.96, '--', 'Color', [.5 .5 .5], 'LineWidth', 1, ...
    'Label', 'z = -1.96', 'LabelHorizontalAlignment', 'left', ...
    'FontSize', 9, 'FontName', fontName);
xline(0, '-', 'Color', [.3 .3 .3], 'LineWidth', 0.8);

set(gca, 'YTick', 1:n_valid, 'YTickLabel', ...
    arrayfun(@(x) sprintf('M%d', ids(x)), ord, 'Uni', 0));
xlabel('z-score (streaks test)', 'FontName', fontName, 'FontSize', 11);
title('Streaks-test z-scores', 'FontName', fontName, 'FontWeight', 'bold', 'FontSize', 13);
text(min(zvals)*0.5, n_valid + 0.4, ...
    sprintf('All z < -17\nFisher p < 10^{-100}'), ...
    'FontName', fontName, 'FontSize', 10, 'VerticalAlignment', 'top');
set(gca, 'Box', 'off', 'TickDir', 'out', 'FontName', fontName, 'FontSize', 10);

sp_sumB = sub2ind_span(nRow_total, nCol_total, nRow_total, floor(nCol_total/2)+1:nCol_total);
ax_zB = subplot(nRow_total, nCol_total, sp_sumB);
hold on;

obs_streaks = [results.n_streaks];
exp_streaks = [results.E_streaks];

for i = 1:n_valid
    plot([obs_streaks(i) exp_streaks(i)], [i i], '-', 'Color', [.7 .7 .7], 'LineWidth', 1.5);
    plot(exp_streaks(i), i, 'o', 'MarkerSize', 6, ...
        'MarkerEdgeColor', [.5 .5 .5], 'MarkerFaceColor', 'none', 'LineWidth', 1.5);
    plot(obs_streaks(i), i, 'o', 'MarkerSize', 6, ...
        'MarkerFaceColor', col_neg, 'MarkerEdgeColor', 'none');
end

set(gca, 'YTick', 1:n_valid, 'YTickLabel', ...
    arrayfun(@(x) sprintf('M%d', results(x).id), 1:n_valid, 'Uni', 0));
xlabel('Number of streaks', 'FontName', fontName, 'FontSize', 11);
title('Observed vs expected streaks', 'FontName', fontName, 'FontWeight', 'bold', 'FontSize', 13);
legend({'', 'Expected (H_0)', 'Observed'}, 'Location', 'se', ...
    'FontName', fontName, 'FontSize', 9, 'Box', 'off');
set(gca, 'Box', 'off', 'TickDir', 'out', 'FontName', fontName, 'FontSize', 10);


outPath = fullfile('', 'Amo_et_al_analysis');

set(fig, 'Units', 'pixels');
set(fig, 'PaperPositionMode', 'auto');

try
    print(fig, '-dsvg', '-painters', [outPath '.svg']);
    
catch ME_svg
    fprintf('\nSVG export failed: %s\n', ME_svg.message);
end

try
    exportgraphics(fig, [outPath '.pdf'], ...
        'ContentType', 'vector', ...
        'BackgroundColor', 'white');
    fprintf('Saved PDF: %s.pdf\n', outPath);
catch
    try
        print(fig, '-dpdf', '-painters', '-loose', [outPath '.pdf']);
    catch ME_pdf
        fprintf('PDF export failed: %s\n', ME_pdf.message);
    end
end

%% print stats
fprintf('\n================================================================\n');
fprintf('  RESIDUAL STREAKS TEST — SUMMARY\n');
fprintf('================================================================\n');
fprintf('%-8s %6s %8s %8s %10s\n', 'Animal', 'Streaks', 'E[Streaks]', 'z', 'p');
fprintf('%s\n', repmat('-', 1, 46));
for vi = 1:n_valid
    r = results(vi);
    fprintf('M%-6d %6d %8.0f %8.2f %10.1e\n', ...
        r.id, r.n_streaks, r.E_streaks, r.z_streaks, r.p_streaks);
end
fprintf('%s\n', repmat('-', 1, 46));
fprintf('Fisher combined:  chi2 = %.1f   df = %d   p = %.1e\n', ...
    chi2_fisher, df_fisher, p_fisher);


%% helper functions

function idx = sub2ind_span(nR, nC, row, cols) %#ok<INUSL>
    idx = (row-1)*nC + cols;
end

function ep = fitExpToData(t, p)
    t = t(:); p = p(:);
    Tinfg = median(p(t >= quantile(t, 0.85)));
    T0g   = median(p(t <= max(5, quantile(t, 0.05))));
    if isnan(Tinfg), Tinfg = min(p); end
    if isnan(T0g),   T0g   = max(p); end
    Tinfg = max(0, Tinfg);
    res = p - Tinfg;
    ok  = res > 1;
    if sum(ok) >= 5
        c     = [ones(sum(ok),1), t(ok)] \ log(res(ok));
        T0g   = max(T0g, exp(c(1)) + Tinfg);
        betag = max(1e-5, -c(2));
    else
        betag = 0.005;
    end
    modelfun = @(b, x) b(2) + (b(1)-b(2)) .* exp(-b(3).*x);
    opts = optimset('Display','off','MaxIter',5000,'TolX',1e-8,'TolFun',1e-9);
    try
        coef  = lsqcurvefit(modelfun, [T0g, Tinfg, betag], t, p, ...
                             [0, 0, 0], [5000, 5000, 2], opts);
        ep.T0   = coef(1);
        ep.Tinf = max(0, coef(2));
        ep.beta = max(1e-5, coef(3));
    catch
        ep.T0 = T0g; ep.Tinf = Tinfg; ep.beta = betag;
    end
    ep.nTrials = max(t);
end

function s = tern(cond, a, b)
    if cond, s = a; else, s = b; end
end
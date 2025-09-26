function [fc, fslope, fintercept] = ComputeFreqFeaturesV2(rf_alines_tmp, rf_freq_ref, varargin)
% ComputeFreqFeatures  Extracts center frequency and linear fit features from RF A-lines.
%
%   [fc, fslope, fintercept] = ComputeFreqFeatures(rf_alines_tmp, rf_freq_ref)
%   computes:
%       - fc: center frequency (MHz) — the frequency at which the normalized
%             spectral amplitude is maximal in the 1..50 MHz band
%       - fslope, fintercept: slope and intercept of a linear fit to the
%             normalized spectrum in a midband window around sensitivity freq
%
%   Inputs:
%       rf_alines_tmp : (Nz x Nalines) matrix of RF A-lines (columns are A-lines)
%       rf_freq_ref   : vector giving frequency reference magnitude for Nz_ref bins
%
%   Optional name-value pairs:
%       'Plot'        : true/false (default false) => show diagnostic plots
%       'Sensitivity' : scalar center (MHz) for midband fit (default 20)
%       'Halfband'    : half-width (MHz) for midband fit (default 8)
%
%   Notes:
%       - Assumes rf_freq_ref corresponds to positive-frequency bins up to Nz_ref.
%       - If rf_alines_tmp has fewer rows than 2*Nz_ref, zero-padding is applied.

% Parse optional args
p = inputParser;
addParameter(p, 'Plot', false, @(x) islogical(x) || (isnumeric(x) && isscalar(x)));
addParameter(p, 'Sensitivity', 20, @isnumeric);
addParameter(p, 'Halfband', 8, @isnumeric);
parse(p, varargin{:});
plot_flag = logical(p.Results.Plot);
f_sensitivity_c = p.Results.Sensitivity;
f_halfband = p.Results.Halfband;

% Validate inputs
if nargin < 2
    error('ComputeFreqFeatures requires rf_alines_tmp and rf_freq_ref as inputs.');
end
if isempty(rf_alines_tmp) || isempty(rf_freq_ref)
    error('Inputs must not be empty.');
end

[Nz, Nalines] = size(rf_alines_tmp);
Nz_ref = numel(rf_freq_ref);

% Ensure we have at least 2*Nz_ref rows by zero-padding if necessary
if Nz < 2 * Nz_ref
    pad_rows = 2 * Nz_ref - Nz;
    rf_alines_aug = [rf_alines_tmp; zeros(pad_rows, Nalines)];
else
    rf_alines_aug = rf_alines_tmp;
end

% Normalize reference frequency vector to unit max (prevent divide-by-zero)
rf_freq_ref = rf_freq_ref(:);
if max(abs(rf_freq_ref)) == 0
    warning('rf_freq_ref has zero maximum; using ones to avoid division by zero.');
    rf_freq_ref = ones(size(rf_freq_ref));
else
    rf_freq_ref = rf_freq_ref / max(abs(rf_freq_ref));
end

% Compute magnitude spectrum averaged across A-lines.
% We compute the one-sided positive-frequency content by summing symmetric bins.
A = fft(rf_alines_aug, [], 1);                 % FFT along rows (depth)
A = abs(A);                                    % magnitude
% Sum symmetric positive-frequency bins: take first Nz_ref rows and add mirrored bins (if any)
pos_part = A(1:Nz_ref, :);
if size(A,1) >= 2 * Nz_ref
    mirrored = A(Nz_ref+1:2*Nz_ref, :);
    pos_part = pos_part + flipud(mirrored);   % small allocation but simple and readable
end
spec_mean = mean(pos_part, 2);                 % mean across columns => (Nz_ref x 1)

% Normalize by rf_freq_ref and scale to unit max
spec_norm = spec_mean(:) ./ rf_freq_ref(:);    % (Nz_ref x 1)
spec_norm = spec_norm / max(spec_norm);        % guard: rf_freq_ref nonzero as above

% Crop to 1..50 MHz band: assume freq_list maps to 1..50 over Nz_ref/2 bins
half_len = floor(Nz_ref/2);
spec_50 = spec_norm(1:half_len);

% frequency vector for those bins (exclude 0): linspace(1,50,half_len+1) with first removed
freq_list = linspace(1, 50, half_len + 1);
freq_list = freq_list(2:end);                  % length half_len

% center frequency: index of maximal spectral magnitude in 1..50 MHz band
[~, idx_max] = max(spec_50);
fc = freq_list(idx_max);

% Prepare midband window indices for linear fit
lower_bound = f_sensitivity_c - f_halfband;
upper_bound = f_sensitivity_c + f_halfband;
% find first indices greater than bounds (safe fallback to endpoints)
idx_lower = find(freq_list > lower_bound, 1, 'first');
if isempty(idx_lower), idx_lower = 1; end
idx_upper = find(freq_list > upper_bound, 1, 'first');
if isempty(idx_upper), idx_upper = length(freq_list); end

% Clip to reasonable neighborhood (ensure at least two points)
if idx_upper <= idx_lower
    idx_upper = min(idx_lower + 1, length(freq_list));
end

freq_midband = freq_list(idx_lower:idx_upper);
spec_midband = spec_50(idx_lower:idx_upper);

% Linear least-squares fit (intercept + slope)
% X * b = y  => b = X \ y
X = [ones(numel(freq_midband),1), freq_midband(:)];
b = X \ spec_midband(:);
fintercept = b(1);
fslope = b(2);

if plot_flag
    figure;
    yyaxis left
    plot(spec_mean(1:half_len));
    ylabel('Magnitude (mean)');
    yyaxis right
    plot(spec_50);
    hold on;
    plot(freq_midband, fintercept + fslope * freq_midband, '--', 'LineWidth', 1.5);
    title(sprintf('Center freq = %.2f MHz', fc));
    xlabel('Index / Frequency bin');
    legend('spec mean','spec normalized (1-50 MHz)','linear fit');
    hold off;
end

end

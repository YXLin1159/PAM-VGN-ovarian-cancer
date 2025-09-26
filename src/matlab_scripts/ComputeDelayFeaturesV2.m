function [t1_med, t2_med, t1_all, t2_all] = ComputeDelayFeaturesV2(rf_alines, varargin)
% ComputeDelayFeatures  Compute delay times to first RF peak (t1) and to first
% envelope peak (t2) per A-line. t2 is now the first peak of the Hilbert
% envelope of the interpolated RF A-line.
%
%   [t1_med, t2_med] = ComputeDelayFeatures(rf_alines)
%   [t1_med, t2_med, t1_all, t2_all] = ComputeDelayFeatures(..., 'Param',Value,...)
%
% Optional Name-Value pairs (defaults in parentheses):
%   'FilterOrder'      (64)   - Length for fir2 HF cancellation filter
%   'InterpFactor'     (2)    - Interpolation factor along depth
%   'SmoothWin'        (5)    - movmean smoothing window
%   'DerivKernel'      ([0.10455 0.292315 0 -0.292315 -0.10455])
%   'ThreshScale'      (1.25) - threshold multiplier for candidate detection
%   'MinThreshold'     (0.1)
%   'SampleIntervalNS' (2.5)  - ns per sample after interpolation
%   'UseParfor'        (false)
%
% Outputs:
%   t1_med, t2_med - median t1/t2 across A-lines (units: ns)
%   t1_all, t2_all - per-A-line values (NaN when undefined)

% Parse optional inputs
p = inputParser;
addParameter(p, 'FilterOrder', 64, @(x) isnumeric(x) && isscalar(x));
addParameter(p, 'InterpFactor', 2, @(x) isnumeric(x) && isscalar(x));
addParameter(p, 'SmoothWin', 5, @(x) isnumeric(x) && isscalar(x));
addParameter(p, 'DerivKernel', [0.104550  0.292315  0.000000 -0.292315 -0.104550], @isnumeric);
addParameter(p, 'ThreshScale', 1.25, @isnumeric);
addParameter(p, 'MinThreshold', 0.1, @isnumeric);
addParameter(p, 'SampleIntervalNS', 2.5, @isnumeric);
addParameter(p, 'UseParfor', false, @(x) islogical(x) || isnumeric(x));
parse(p, varargin{:});
opts = p.Results;

% Validate input
if isempty(rf_alines) || ndims(rf_alines) ~= 2
    error('rf_alines must be a non-empty 2D matrix (Nz x N_alines).');
end

% Precompute HF cancellation filter (same behavior as original)
f = [0 0.5 0.5 1];
m = [1 0.9 0.1 0];
hf_filter = fir2(opts.FilterOrder, f, m);

[Nz, Nlines] = size(rf_alines);
Nz_interp = Nz * opts.InterpFactor;

% Preallocate outputs (NaN indicates undefined)
t1_all = nan(Nlines,1);
t2_all = nan(Nlines,1);

% Derivative kernel
d = opts.DerivKernel(:)';

% Interpolation grid
xi = linspace(1, Nz, Nz_interp);

% Determine if signal processing functions are available
has_findpeaks = exist('findpeaks','file') == 2;   % requires Signal Processing Toolbox
has_hilbert = exist('hilbert','file') == 2;

% Processing loop (optionally parfor)
if opts.UseParfor
    for idx_aline = 1:Nlines
        [t1_val, t2_val] = process_single_aline(rf_alines(:,idx_aline));
        t1_all(idx_aline) = t1_val;
        t2_all(idx_aline) = t2_val;
    end
else
    for idx_aline = 1:Nlines
        [t1_val, t2_val] = process_single_aline(rf_alines(:,idx_aline));
        t1_all(idx_aline) = t1_val;
        t2_all(idx_aline) = t2_val;
    end
end

% Return medians ignoring NaNs
t1_med = nanmedian(t1_all);
t2_med = nanmedian(t2_all);

%% Nested helper
    function [t1_out, t2_out] = process_single_aline(aline)
        % Returns t1 and t2 for a single A-line (NaN if undefined)
        t1_out = NaN;
        t2_out = NaN;

        % 1) Smooth + HF cancellation filter
        aline_smooth = movmean(aline, opts.SmoothWin);
        aline_f = conv(aline_smooth, hf_filter, 'same');

        % 2) Interpolate
        aline_interp = interp1(1:Nz, aline_f, xi, 'pchip'); % pchip robust alternative to 'v5cubic'
        max_abs_aline = max(abs(aline_interp));
        if max_abs_aline == 0
            return; % no signal
        end
        aline_interp = aline_interp / max_abs_aline;

        % 3) derivatives and filtering
        aline_deriv = conv(aline_interp, d, 'same');
        aline_deriv = medfilt1(aline_deriv);
        max_deriv = max(abs(aline_deriv));
        if max_deriv ~= 0
            aline_deriv = aline_deriv / max_deriv;
        end

        aline_deriv2 = conv(aline_deriv, d, 'same');
        aline_deriv2 = medfilt1(aline_deriv2, 5);
        max_deriv2 = max(abs(aline_deriv2));
        if max_deriv2 ~= 0
            aline_deriv2 = aline_deriv2 / max_deriv2;
        end

        % 4) threshold for peak candidate detection (based on early-depth energy)
        window_len = min(30, length(aline_interp));
        threshold_tmp = max(opts.ThreshScale * max(abs(aline_interp(1:window_len))), opts.MinThreshold);

        % t1 detection: same as before (first RF peak candidate)
        candidates = find(abs(aline_interp) > threshold_tmp & abs(aline_deriv) < threshold_tmp);
        if isempty(candidates)
            return;
        end
        % first contiguous candidate => t1
        t1_idx = candidates(1);
        % find first non-contiguous candidate after the first run for t2_old (kept for fallback)
        t2_idx_old = NaN;
        for k = 2:numel(candidates)
            if candidates(k) - candidates(k-1) > 1
                t2_idx_old = candidates(k);
                break;
            end
        end

        % Now compute t2 using Hilbert envelope peak (preferred)
        t2_idx = NaN;
        if has_hilbert
            env = abs(hilbert(aline_interp));   % analytic envelope
            % normalize envelope to same scaling as aline_interp for threshold comparison
            if max(env) ~= 0
                env = env / max(env);
            end

            if has_findpeaks
                % find all peaks; choose the first peak whose amplitude > threshold_tmp
                [pks, locs] = findpeaks(env);
                if ~isempty(pks)
                    % find first peak in time order with amplitude above threshold
                    idx_valid = find(pks > threshold_tmp, 1, 'first');
                    if ~isempty(idx_valid)
                        t2_idx = locs(idx_valid);
                    else
                        % if no peaks exceed threshold, fallback to first peak in time order
                        t2_idx = locs(1);
                    end
                end
            else
                % if findpeaks is not available, use simple threshold-based search on envelope:
                cand_env = find(env > threshold_tmp);
                if ~isempty(cand_env)
                    % pick the first contiguous run and select its first index as the peak candidate
                    t2_idx = cand_env(1);
                    for kk = 2:numel(cand_env)
                        if cand_env(kk) - cand_env(kk-1) > 1
                            break;
                        else
                            % update to index of maximum within the contiguous run
                            if env(cand_env(kk)) > env(t2_idx)
                                t2_idx = cand_env(kk);
                            end
                        end
                    end
                end
            end
        end

        % Fallback: if Hilbert-based t2 not found, use previous contiguous-candidate method
        if isnan(t2_idx) && ~isnan(t2_idx_old)
            t2_idx = t2_idx_old;
        end

        % If t1 and t2 are found, find baseline t0 similar to original logic
        if ~isnan(t1_idx) && ~isnan(t2_idx)
            % baseline search: last index before t1 where envelope and derivatives are below threshold
            search_range = 1:max(1, t1_idx);
            k0 = find(abs(aline_interp(search_range)) < threshold_tmp & ...
                      abs(aline_deriv(search_range)) < threshold_tmp & ...
                      abs(aline_deriv2(search_range)) < threshold_tmp);
            if ~isempty(k0)
                t0_idx = max(k0) - 1;
                if t0_idx < 1
                    t0_idx = 1;
                end
            else
                t0_idx = 1;
            end

            % convert to times (ns)
            t1_out = (t1_idx - t0_idx) * opts.SampleIntervalNS;
            t2_out = (t2_idx - t0_idx) * opts.SampleIntervalNS;
        end
    end

end

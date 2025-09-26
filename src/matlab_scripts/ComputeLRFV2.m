function [segbv, idx_orient] = computeLRFV2(mip_neighbor)
% COMPUTELRF Computes local radial features and vessel orientation from MIP images
%
% Inputs:
%   mip_neighbor - H x W image (neighboring MIP)
%
% Outputs:
%   segbv        - Segmented vessel map
%   idx_orient   - Orientation map (deg) for each pixel

%% Parameters
[H, W] = size(mip_neighbor);
rho_1 = 10;                 % Long grain
rho_2_test = 2:10;          % Short grain (vessel thickness)
rho_3 = 2;                   % Blurring kernel
N_angle = 36;                % Number of orientations
angle_list = linspace(90, -90, N_angle+1);
angle_list(end) = [];        % Remove last duplicate
x1 = -2*rho_1:2*rho_1;
x3 = -2*rho_3:2*rho_3;

% Precompute kernels
K_rho3 = (1/rho_3) * exp(-abs(x3).^2 / (2*rho_3^2));
K_rho3 = K_rho3' * K_rho3;
k  = [0.030320  0.249724  0.439911  0.249724  0.030320];
d  = [0.104550  0.292315  0.000000 -0.292315 -0.104550];
K_rho2 = conv(conv(k, d), d);  % 2nd derivative

% Preprocess MIP image
test_img_pre = adapthisteq(imbilatfilt(mat2gray(mip_neighbor), 0.5, 2));

%% Local Radon Transform
N_thickness = numel(rho_2_test);
LRT_all = zeros(H, W, N_angle, N_thickness);

for idx_rho2 = 1:N_thickness
    rho_2 = rho_2_test(idx_rho2);
    y2 = linspace(-2*rho_2, 2*rho_2, numel(K_rho2));
    [XX, YY] = meshgrid(x1, y2);
    XY = [XX(:), YY(:)];

    for idx_angle = 1:N_angle
        theta = angle_list(idx_angle);
        R = [cosd(theta), -sind(theta); sind(theta), cosd(theta)];
        rotXY = XY * R;

        XXqr = reshape(rotXY(:,1), size(XX));
        YYqr = reshape(rotXY(:,2), size(YY));

        K_rho1 = (1/rho_1) * exp(-XXqr.^2 / (2*rho_1^2));
        K_rho2_interp = interp1(y2, K_rho2, YYqr, 'v5cubic');
        K_rho2_interp(isnan(K_rho2_interp)) = 0;

        h_eta = K_rho1 .* K_rho2_interp;
        e_eta = conv2(K_rho3, h_eta, 'same');

        % Pad and convolve
        test_img_pad = padarray(test_img_pre, [2*rho_1, 2*rho_1], 'replicate', 'both');
        conv_theta = -conv2(test_img_pad, e_eta, 'same');
        conv_theta(conv_theta < 0) = 0;

        LRT_all(:, :, idx_angle, idx_rho2) = conv_theta(2*rho_1+1:end-2*rho_1, 2*rho_1+1:end-2*rho_1);
    end
end

% Maximum over thickness dimension
LRT_all = squeeze(max(LRT_all, [], 4));

%% Compute orientation map
idx_orient = zeros(H, W);
angle_template = linspace(180, -180, 2*N_angle+1);
angle_template(end) = [];
cosine_template = cosd(angle_template);
cosine_template(cosine_template < 0) = 0;

for idx_x = 1:H
    for idx_y = 1:W
        test_vec = medfilt1(squeeze(LRT_all(idx_x, idx_y, :)));
        [r, lag] = xcorr(test_vec, cosine_template, N_angle);
        [~, max_idx] = max(r);
        idx_orient(idx_x, idx_y) = -(-lag(max_idx)*180/N_angle - 90);
    end
end

%% Segment vessel map
segbv = squeeze(max(LRT_all, [], 3));
segbv = adapthisteq(mat2gray(imbilatfilt(segbv, 0.1, 1)));

end

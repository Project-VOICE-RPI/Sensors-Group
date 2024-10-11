%% GSR Signal Preprocessing and Continuous Decomposition Analysis
% This script performs preprocessing on raw GSR data and applies
% Continuous Decomposition Analysis using non-negative deconvolution
% directly on the preprocessed signal.

clear; close all; clc;

%% Section 1: Generating A Raw GSR Signal

fs_raw = 1000;
duration = 300;
t_raw = 0:1/fs_raw:duration;

tonic_trend = 0.5 + 0.0001 * t_raw;

% Generate phasic components
% Simulate SCRs as exponentially decaying peaks at random intervals
num_scrs = 50;
scr_times = sort(rand(1, num_scrs) * duration);
scr_amplitudes = 0.05 + 0.1 * rand(1, num_scrs);
scr_tau = 1;

phasic_signal = zeros(size(t_raw));

for i = 1:num_scrs
    idx = find(t_raw >= scr_times(i), 1);
    scr_length = ceil(5 * fs_raw);
    scr_time_vector = (0:scr_length - 1) / fs_raw;
    scr_waveform = scr_amplitudes(i) * exp(-scr_time_vector / scr_tau);
    end_idx = min(idx + scr_length - 1, length(phasic_signal));
    phasic_signal(idx:end_idx) = phasic_signal(idx:end_idx) + scr_waveform(1:(end_idx - idx +1));
end

raw_gsr_signal = tonic_trend + phasic_signal;

noise_level = 0.005;
raw_gsr_signal = raw_gsr_signal + noise_level * randn(size(raw_gsr_signal));

% Plot the raw GSR signal
figure;
plot(t_raw, raw_gsr_signal);
title('Raw GSR Signal (Synthetic)');
xlabel('Time (s)');
ylabel('Skin Conductance (\muS)');

%% Section 2: Preprocessing Steps

cutoff_freq = 5;
filter_order = 4;

Wn = cutoff_freq / (fs_raw / 2);

[b, a] = butter(filter_order, Wn, 'low');

filtered_signal = filtfilt(b, a, raw_gsr_signal);

fs_new = 10;
downsample_factor = fs_raw / fs_new;

if mod(downsample_factor, 1) ~= 0
    error('The downsampling factor must be an integer. Adjust fs_new accordingly.');
end

downsampled_signal = downsample(filtered_signal, downsample_factor);
t_downsampled = downsample(t_raw, downsample_factor);

window_size = 5;
smoothed_signal = movmean(downsampled_signal, window_size);

% Plot the preprocessed GSR signal
figure;
plot(t_downsampled, smoothed_signal);
title('Preprocessed GSR Signal');
xlabel('Time (s)');
ylabel('Skin Conductance (\muS)');

%% Section 3: Continuous Decomposition Analysis

t = t_downsampled;
tau1 = 0.75;
tau2 = 2;

irf_duration = 10;
irf_time = 0:1/fs_new:irf_duration;
irf = ( (1 / (tau1 - tau2)) * (exp(-irf_time / tau1) - exp(-irf_time / tau2)) );
irf = irf / max(irf);

signal_for_deconvolution = smoothed_signal;

N = length(signal_for_deconvolution);
M = length(irf);
K_full = convmtx(irf', N);
K = K_full(1:N, :);

lambda = 0.1;

% Set up NNLS problem
options = optimoptions('lsqlin', 'Algorithm', 'interior-point', 'Display', 'off');

driver_estimate = lsqlin(K, signal_for_deconvolution', [], [], [], [], zeros(N,1), [], [], options);

% Convolve the estimated driver with the IRF to get the phasic component
phasic_component_full = conv(driver_estimate, irf);
phasic_component = phasic_component_full(1:N);

tonic_component = smoothed_signal - phasic_component';

% Plot the estimated driver function
figure;
plot(t, driver_estimate);
title('Estimated Driver Function (Non-negative Deconvolution)');
xlabel('Time (s)');
ylabel('Amplitude');

% Plot the phasic and tonic components
figure;
subplot(3,1,1);
plot(t, smoothed_signal);
title('Preprocessed GSR Signal');
xlabel('Time (s)');
ylabel('Skin Conductance (\muS)');

subplot(3,1,2);
plot(t, tonic_component, 'r');
title('Tonic Component');
xlabel('Time (s)');
ylabel('Skin Conductance (\muS)');

subplot(3,1,3);
plot(t, phasic_component, 'g');
title('Phasic Component');
xlabel('Time (s)');
ylabel('Skin Conductance (\muS)');

% Significant driver events
threshold = mean(driver_estimate) + std(driver_estimate);
[driver_peaks, locs] = findpeaks(driver_estimate, 'MinPeakHeight', threshold);

% Plot the driver function with detected peaks
figure;
plot(t, driver_estimate);
hold on;
plot(t(locs), driver_estimate(locs), 'rv', 'MarkerFaceColor', 'r');
title('Estimated Driver Function with Detected Peaks');
xlabel('Time (s)');
ylabel('Amplitude');
legend('Driver Function', 'Peaks');
hold off;
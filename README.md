# Efficient-Signal-Detection-MIMO-NOMA-QAOA
Final Year Project - Efficient Signal Detection in MIMO-NOMA Systems Using QAOA for 6G Wireless Communication
clc; clear; close all;
%% ================= SYSTEM PARAMETERS =================
MIMO_sizes = [2 4 8]; % MIMO configurations
SNR_dB = -10:5:20;
numBits = 2000;
BER_QAOA = zeros(length(MIMO_sizes), length(SNR_dB));
BER_ML = zeros(length(MIMO_sizes), length(SNR_dB));
BER_SIC = zeros(length(MIMO_sizes), length(SNR_dB));
%% ================= MAIN SIMULATION =================
for m = 1:length(MIMO_sizes)
Nt = MIMO_sizes(m);
Nr = Nt;
powerAlloc = ones(Nt,1)/Nt;
for snrIdx = 1:length(SNR_dB)
snr = 10^(SNR_dB(snrIdx)/10);
noiseVar = 1/snr;
errQ = 0; errM = 0; errS = 0;
for n = 1:numBits
% -------- BPSK symbols --------
bits = randi([0 1], Nt, 1);
s = 2*bits - 1;
% -------- Rayleigh channel --------
H = (randn(Nr,Nt) + 1j*randn(Nr,Nt)) / sqrt(2);
% -------- Transmission --------
x = sqrt(powerAlloc).*s;
noise = sqrt(noiseVar/2)*(randn(Nr,1)+1j*randn(Nr,1));
y = H*x + noise;
% -------- QAOA Detection --------
sQ = quantum_qaoa_detector(y,H,powerAlloc);
errQ = errQ + sum(sQ~=s);
% -------- ML Detection --------
sM = ml_detector(y,H,Nt);
errM = errM + sum(sM~=s);
% -------- SIC Detection --------
sS = sic_detector(y,H,Nt);
errS = errS + sum(sS~=s);
end
BER_QAOA(m,snrIdx) = errQ/(numBits*Nt);
BER_ML(m,snrIdx) = errM/(numBits*Nt);
BER_SIC(m,snrIdx) = errS/(numBits*Nt);
fprintf('%dx%d | SNR=%d | QAOA=%e | ML=%e | SIC=%e\n',...
Nt,Nt,SNR_dB(snrIdx),...
BER_QAOA(m,snrIdx), BER_ML(m,snrIdx), BER_SIC(m,snrIdx));
end
end
%% ================= BER PLOT =================
figure;
style = {'-o','-s','-^'};
for m = 1:length(MIMO_sizes)
semilogy(SNR_dB, BER_QAOA(m,:), style{m}, 'LineWidth',2); hold on;
semilogy(SNR_dB, BER_ML(m,:), '--', 'LineWidth',2);
semilogy(SNR_dB, BER_SIC(m,:), ':', 'LineWidth',2);
end
grid on;
xlabel('SNR (dB)');
ylabel('BER');
legend('QAOA 2x2','ML 2x2','SIC 2x2',...
'QAOA 4x4','ML 4x4','SIC 4x4',...
'QAOA 8x8','ML 8x8','SIC 8x8',...
'Location','southwest');
title('BER vs SNR');
%% ================= THROUGHPUT =================
Throughput_QAOA = zeros(size(BER_QAOA));
Throughput_ML = zeros(size(BER_ML));
Throughput_SIC = zeros(size(BER_SIC));
for m = 1:length(MIMO_sizes)
Nt = MIMO_sizes(m);
Throughput_QAOA(m,:) = Nt * (1 - BER_QAOA(m,:));
Throughput_ML(m,:) = Nt * (1 - BER_ML(m,:));
Throughput_SIC(m,:) = Nt * (1 - BER_SIC(m,:));
end
figure;
for m = 1:length(MIMO_sizes)
plot(SNR_dB, Throughput_QAOA(m,:), style{m}, 'LineWidth',2); hold on;
plot(SNR_dB, Throughput_ML(m,:), '--', 'LineWidth',2);
plot(SNR_dB, Throughput_SIC(m,:), ':', 'LineWidth',2);
end
grid on;
xlabel('SNR (dB)');
ylabel('Throughput (bits/s/Hz)');
legend('QAOA 2x2','ML 2x2','SIC 2x2',...
'QAOA 4x4','ML 4x4','SIC 4x4',...
'QAOA 8x8','ML 8x8','SIC 8x8',...
'Location','southeast');
title('Throughput vs SNR');
%% ================= COMPLEXITY ANALYSIS =================
Nt_range = 2:10;
p = 5; % QAOA depth
Complexity_ML = 2.^Nt_range; % O(2^Nt)
Complexity_SIC = Nt_range.^3; % O(Nt^3)
Complexity_QAOA = p * Nt_range.^2; % O(p Nt^2)
figure;
semilogy(Nt_range, Complexity_ML, '-o', ...
Nt_range, Complexity_SIC, '-s', ...
Nt_range, Complexity_QAOA, '-^', 'LineWidth',2);
grid on;
xlabel('Number of Transmit Antennas (N_t)');
ylabel('Relative Computational Complexity');
legend('ML','SIC','QAOA','Location','northwest');
title('Computational Complexity Comparison');
%% ======================================================
%% ======================= FUNCTIONS ====================
%% ======================================================
%% -------- QAOA DETECTOR --------
function sHat = quantum_qaoa_detector(y,H,powerAlloc)
Nt = size(H,2);
Heff = H*diag(sqrt(powerAlloc));
Q = real(Heff'*Heff);
c = -2*real(Heff'*y);
s = sign(randn(Nt,1));
gamma = [0.3 0.5 0.7];
beta = [0.6 0.4 0.2];
for k = 1:length(gamma)
s = s - gamma(k)*(Q*s + c);
s = cos(beta(k))*s + sin(beta(k))*sign(randn(Nt,1));
s = sign(real(s));
end
sHat = s;
end
%% -------- ML DETECTOR --------
function sHat = ml_detector(y,H,Nt)
comb = dec2bin(0:2^Nt-1)-'0';
comb = 2*comb-1;
minErr = inf;
sHat = zeros(Nt,1);
for k = 1:size(comb,1)
x = comb(k,:)';
err = norm(y-H*x)^2;
if err < minErr
minErr = err;
sHat = x;
end
end
end
%% -------- SIC DETECTOR (ERROR-FREE) --------
function sHat = sic_detector(y,H,Nt)
sHat = zeros(Nt,1);
H_rem = H;
y_rem = y;
for k = 1:Nt
W = pinv(H_rem);
s_est = W(1,:) * y_rem;
sHat(k) = sign(real(s_est));
y_rem = y_rem - H_rem(:,1)*sHat(k);
H_rem(:,1) = [];
end
end

%%--------------------------------------------------------------------------
% References:
% [1] Tinghe Yan, Qiang Guo,
% Tensor Robust Principal Component Analysis via Weighted Tensor Schatten p
% norm and lp norm, Submit draft, 2023.
% [2] Canyi Lu, Jiashi Feng, Yudong Chen, Wei Liu, Zhouchen Lin, and Shuicheng Yan, 
%Tensor Robust Principal Component Analysis with A New Tensor Nuclear Norm, TPAMI, 2019.
% [3] Quanxue Gao, Pu Zhang, Wei Xia, Deyan Xie, Xinbo Gao, and Dacheng Tao,
%Enhanced Tensor RPCA and its Application, TPAMI, 2021.
%
% version 1.0 -- 23/3/2023
%
%%--------------------------------------------------------------------------
% Written by Tinghe Yan
%%--------------------------------------------------------------------------
%% Load Setting
clc;
clear all;
addpath(genpath(cd));
addpath([pwd, '\metrics']);
addpath([pwd, '\ptrpca']);
%% Load image
pic_name = './data/test_img.png';  
X = double(imread(pic_name));   % original samples
X = X/255;                      % normaliza X to [0,1].
%% Generage corrupted image
maxP = max(abs(X(:)));
[n1,n2,n3] = size(X);
Xn = X;
rhos = 0.1;                       % setting corrupted level eta=10 
ind = find(rand(n1*n2*n3,1)<rhos);
Xn(ind) = rand(length(ind),1);    % noisy samples      
[n1,n2,n3] = size(X);    % sample size:481X321X3 or 321X481X3
n=min(n1,n2);
%% Hyper-Parameters
% Weighted vector of weighted tensor Schatten p-norm
w = [];
% parameter w
w = [w; 1*ones(10,1)];
w = [w; 1.1*ones(70,1)];
w = [w; 1.5*ones(n-80,1)];
% The power p1 of weighted tensor Schatten p-norm, p1 in (0,1]
%  parameter p1:
p1=0.8;
% The power p2 of lp-norm, p2 in (0,1]
% parameter p2:
p2=0.6;

lambda = 1/sqrt(max(n1,n2)*n3);
opts.mu = 1e-4;
opts.tol = 1e-8;
opts.rho = 1.1;
opts.max_iter = 500;
opts.DEBUG = 0;
%% Run pTRPCA
tic;
[L, E, obj, err, iter]  = ptrpca(Xn, lambda, w, p1,p2,opts);
toc;
L = max(L,0);
L = min(L,maxP);
%% Calculate PSNR, SSIM, FSIM, ERGAS, MSAM
L = L*255;
X = X*255;
[psnr, ssim, fsim, ergas, msam]=MSIQA(X,L);
%% Record Results
fid = fopen('test.txt','a+');
fprintf(fid,'The results:');
fprintf(fid,'%15s', ['PSNR£º', num2str(psnr)], ['SSIM£º', num2str(ssim)],...
['FSIM:' ,num2str(fsim)],['ERGAS:' ,num2str(ergas)],['MSAM:',num2str(msam)]);
fprintf(fid,'\n');
%% Compute difference image
diff_img=L-X;
%% Visualization
figure(1);
subplot(2,2,1);
imshow(X/max(X(:)));              % original image
subplot(2,2,2);
imshow(Xn/max(Xn(:)));            % corrupted image
subplot(2,2,3);
imshow(L/max(L(:)));              % recovered image
subplot(2,2,4);
imshow(diff_img);                 % difference image

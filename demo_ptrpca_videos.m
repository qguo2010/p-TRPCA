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
clc;
clear all;
addpath(genpath(cd));
addpath([pwd, '\ptrpca']);
%% Load dataset
dataList={'./data/HighwayI'};
dataName=dataList{1};
load(dataName);  % newdata with size: 4800X3X440
Xn=double(newdata);
[n1,n2,n3] = size(Xn); 
 X=zeros(60,80,n3); % X with size:60X80X440
 B=X;
 S=X;
%% Hyper-Parameters
% Weighted vector of weighted tensor Schatten p-norm
w = [];
% parameter w
n=min(n1,n2);
w = [w; 1*ones(n,1)];
% The power p1 of weighted tensor Schatten p-norm, p1 in (0,1]
%  parameter p1：
p1=1;
% The power p2 of lp-norm, p2 in (0,1]
% parameter p2：
p2=0.7;
lambda = 1/sqrt(max(n1,n2)*n3);
opts.mu = 1e-6;
opts.tol = 1e-8;
opts.rho = 1.1;
opts.max_iter = 500;
opts.DEBUG = 0;
%% Run pTRPCA
tic
[B1, S1, obj, err, iter]  = ptrpca(Xn, lambda, w, p1,p2);
toc
%% reshape tensor
k=0;
for i=1:n3
   for j=1:3
        k=k+1;
       X(:,:,k)=reshape(Xn(:,j,i),60,[]); 
       B(:,:,k)=reshape(B1(:,j,i),60,[]); 
       S(:,:,k)=reshape(S1(:,j,i),60,[]); 
   end
end
last_img=X(:,:,[k-2,k-1,k]);
last_B=B(:,:,[k-2,k-1,k]);
%% Visualization
figure(1);
subplot(1,2,1);
imshow(uint8(last_img));        % original image
subplot(1,2,2);
imshow(uint8(last_B));          % background image

clc;
clear;
close all;
% load image
image = load('MNIST_image.mat');
A = reshape(image.MNIST_image,28,28)';
% divide into blocks, each block size is 32x32
Block=[];
data = [];
inc2 = 0;
flag2=1;
while(flag2 <= 7)
    inc = 0;
    flag=1;
    while(flag <= 7)
        for i = (1+inc2):(4+inc2)
            for j = (1+inc):(4+inc)
                Block = [Block A(i,j)];
            end

        end
        data = [data ; Block]
        Block=[];
        flag = flag+1;
        inc = inc + 4;
    end
    flag2 = flag2 + 1;
    inc2 = inc2 + 4;
end

%% ART
rho=0.95;                            % vigilance parameter
alpha = 1e-3;                   % choice parameter 
beta = 0.7;                       % learning parameter: (0,1] (beta=1: "fast learning")    
W = [];                         % weight vectors
labels = [];                    % class labels
dim = [];                       % original dimension of data set  
nCategories = 0;                % total number of categories
Epoch = 0;                      % current epoch    
T = [];                         % category activation/choice function vector
M = [];                         % category match function vector  
W_old = [];                     % old weight vectors


% Data Information            
[nSamples, dim] = size(data);
labels = zeros(nSamples, 1);

% Normalization and Complement coding
x = data;
x = [x 255-x];
%x = double(x);
% Initialization 
if isempty(W)             
    W = 255*ones(1, 2*dim);                                   
    nCategories = 1;                 
end              
W_old = W; 

Epoch = 0;
flag = 0;
flag2 = 0;
while(true)

    Epoch = Epoch + 1;

    for i=1:nSamples  % loop over samples
        
        if or(isempty(T), isempty(M))
            
            T = zeros(nCategories, 1);     
            M = zeros(nCategories, 1); 
            
            for j=1:nCategories 

                numerator = norm(min(x(i,:), W(j, :)), 1);

                T(j, 1) = numerator/(alpha + norm(W(j, :), 1));

                M(j, 1) = numerator/norm(x(i,:),1);

            end

        end     

        [~, index] = sort(T, 'descend');  % Sort activation function values in descending order                    

        mismatch_flag = true;  % mismatch flag 

        for j=1:nCategories  % loop over categories                       

            bmu = index(j);  % Best Matching Unit 

            if M(bmu) >= rho% Vigilance Check - Pass 

                W(index(j),:) = beta*(min(x(i,:), W(index(j),:)))+ (1-beta)*W(index(j),:);     

                labels(i,j) = bmu;  % update sample labels

                mismatch_flag = false;  % mismatch flag 
                
                break;
            
            end                               

        end
        
        if mismatch_flag  % If there was no resonance at all then create new category

            nCategories = nCategories + 1;  % increment number of categories

            W(nCategories,:) = x(i,:);  % fast commit                         

        end 
        
        T = [];
        M = [];
        
    end
    
    stop = false; 

    if isequal(W, W_old)

        stop = true;
        
    end 
    
    if stop

        break;

    end 

    W_old = W;
    
end 
%% image reconstruction

block_code = reshape(labels(:,1)',7,7)';

temp = [];
comp_img = [];

for i=1:7
    for j=1:7
        read_code_book = reshape(W(block_code(i,j),1:16),4,4)';
        temp = [temp read_code_book];
    end
    comp_img = [comp_img; temp];
    temp = [];
end

% Calculations
Compression_ratio = (28*28)/(49 + nCategories*16);

Error = [];

diff = (comp_img - A);

for i = 1:28
    Error = [Error diff(i,:)];
end

MSE = sum(Error.^2)/(28*28);

[PSNR, SNR] = psnr(A, comp_img, 255);

% display images
movegui(figure,'east');
colormap gray;
imagesc(comp_img);
title('RECONSTRUCTED IMAGE');

movegui(figure,'west');
colormap gray;
imagesc(A);
title('ORIGINAL IMAGE');

% display values
Compression_ratio
MSE
PSNR
SNR

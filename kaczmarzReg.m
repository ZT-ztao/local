function [ x, eval] = kaczmarzReg( A,b,iterations,lambd,shuff,enforceReal,enforcePositive, tolerance, c)
%kaczmarzReg(S_truncated(:,:), u_mean_truncated(:), 1, 1*10^-6, 0, 1, 1);
% regularized Kaczmarz
% As published here: http://iopscience.iop.org/article/10.1088/0031-9155/55/6/003/meta on page 1582.
% Other references : Saunders M 1995: Solution of sparse rectangular
% systems using LSQR and CRAIG
% or
% From Dax A 1993: On row relaxation methods for large constrained 
% least squares problems

% initialization of the variable
[N, M] = size(A);  % N:column  M:row
x = complex(zeros(N,1)); 
residual = complex(zeros(M,1));
rowIndexCycle = 1:M;

x_prev=zeros(N,1);
% calculate the energy of each frequency component
energy = rowEnergy(A);

% may use a randomized Kaczmarz
if shuff
    rowIndexCycle = randperm(M); % Ëæ»úÖÃ»»
end

% estimate regularization parameter
lambdZero = sum(energy.^2)/N;
lambdIter = lambd*lambdZero;
l = 1;
error = 1e5;

while l <= iterations && error >= tolerance
    for m = 1:M
        k = rowIndexCycle(m);
        
        if energy(k) > 0
            tmp = A(:,k).'*x;
            beta = (b(k) - tmp - sqrt(lambdIter)*residual(k)) / (energy(k)^2 + lambdIter);
            x = x + beta*conj(A(:,k));
            residual(k) = residual(k) + beta*sqrt(lambdIter);
        end
    end
    
    if enforceReal && ~isreal(x)
        x = complex(real(x),0);
    end
    
    if enforcePositive
        x(real(x) < 0) = 0;
    end
    
    % iteration
    error = norm(x - x_prev, 2)/(norm(x_prev, 2)+ 1e-3);
    step = norm(x - x_prev, 1);
    x_prev = x; 
    bias = norm(A.' * x - b, 1);
    obj = ( 1/2*norm(A.'*x - b, 2).^2 + lambd*norm(x,2).^2 );
%     fprintf('Iter : %f, step : %f, error : %f, obj : %f \n', l, step, bias, obj);
    fprintf('Iter : %f, error : %f, obj : %f, step : %f \n', l, bias, obj, step);

    l = l + 1;
end
eval.psnr = eval_psnr(c, x);
eval.ssim = eval_ssim(c, x);
eval.nrmse= eval_nrmse(c, x);
% out_bias = norm(A.' * x - b, 2);
% out_solu = norm(x, 2);
fprintf('lambda : %f, psnr  : %f, ssim : %f , nrmse : %f \n', lambd, eval.psnr, eval.ssim, eval.nrmse);
end
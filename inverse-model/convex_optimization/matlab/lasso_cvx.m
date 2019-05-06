

A = load('A.mat');
A = A.A;

b = load('b.mat');
b = b.b;
% b = reshape(b, [32768, 1]);

% x0 = load('x0.mat');
% x0 = x0.x0;

x0 = reshape(x0,[32, 1]);

lambda = 0.001;
rho = 1;
alpha = 1.5;

[z, history] = lasso(A, b, lambda, rho, alpha, x0);

% for i = 0.01: 10000
%     lambda = 1 / i;
%     [z, history] = lasso(A, b, lambda, rho, alpha);
%     err = sum(abs(x0-z));
%     if err <= 1
%         break
%     end
%     fprintf('Error: %d, lambda = %d \r',err, lambda);
% end


figure;
plot(x);
hold on;
plot(x0);

function [z, history] = lasso(A, b, lambda, rho, alpha, x0)
t_start = tic;

QUIET    = 0;
MAX_ITER = 3000;
ABSTOL   = 1e-10;
RELTOL   = 1e-10;

[m, n] = size(A);

% save a matrix-vector multiply
Atb = A'*b;

x = zeros(n,1);
z = zeros(n,1);
u = zeros(n,1);

% cache the factorization
[L U] = factor(A, rho);

% if ~QUIET
%     fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
%       'r norm', 'eps pri', 's norm', 'eps dual', 'objective');
% end
figure;
for k = 1:MAX_ITER
    
    % x-update
    q = Atb + rho*(z - u);    % temporary value
    if( m >= n )    % if skinny
       x = U \ (L \ q);
    else            % if fat
       x = q/rho - (A'*(U \ ( L \ (A*q) )))/rho^2;
    end
    
    if mod(k,100) == 0
        plot(x);
        hold on;
        plot(x0);
        pause(0.1);
    end
    

    % z-update with relaxation
    zold = z;
    x_hat = alpha*x + (1 - alpha)*zold;
    z = shrinkage(x_hat + u, lambda/rho);
     
    plot(x_hat);
    pause(0.1);

    
    % u-update
    u = u + (x_hat - z);

    % diagnostics, reporting, termination checks
    history.objval(k)  = objective(A, b, lambda, x, z);

    history.r_norm(k)  = norm(x - z);
    history.s_norm(k)  = norm(-rho*(z - zold));

    history.eps_pri(k) = sqrt(n)*ABSTOL + RELTOL*max(norm(x), norm(-z));
    history.eps_dual(k)= sqrt(n)*ABSTOL + RELTOL*norm(rho*u);

%     if ~QUIET
%         fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', k, ...
%             history.r_norm(k), history.eps_pri(k), ...
%             history.s_norm(k), history.eps_dual(k), history.objval(k));
%     end

    if (history.r_norm(k) < history.eps_pri(k) && ...
       history.s_norm(k) < history.eps_dual(k))
         break;
    end

end

end
function p = objective(A, b, lambda, x, z)
    p = ( 1/2*sum((A*x - b).^2) + lambda*norm(z,1) );
end

function z = shrinkage(x, kappa)
    z = max( 0, x - kappa ) - max( 0, -x - kappa );
end

function [L U] = factor(A, rho)
    [m, n] = size(A);
    if ( m >= n )    % if skinny
       L = chol( A'*A + rho*speye(n), 'lower' );
    else            % if fat
       L = chol( speye(m) + 1/rho*(A*A'), 'lower' );
    end

    % force matlab to recognize the upper / lower triangular structure
    %L = sparse(L);
    %U = sparse(L');
    U = L';
end



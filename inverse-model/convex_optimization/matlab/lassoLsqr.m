
A = load('A.mat');
A = A.A;
b = load('b.mat');
b = b.b;
x0 = load('x0.mat');
x0 = x0.x0;
x0 = x0';

x_ini = load('x_ini.mat');
x_ini = x_ini.x_ini;
x_ini = x_ini';

lambda = 0.1;
rho = 10;
alpha = 1.5;

[z, history] = lasso_lsqr(A, b, lambda, rho, alpha, x_ini, x0);

figure;
plot(z);
hold on;
plot(x0);
legend('Optimization','GroundTruth')

% for i = 0.01: 10000
%     lambda = 1 / i;
%     [z, history] = lasso(A, b, lambda, rho, alpha);
%     err = sum(abs(x0-z));
%     if err <= 1
%         break
%     end
%     fprintf('Error: %d, lambda = %d \r',err, lambda);
% end




function [z, history] = lasso_lsqr(A, b, lambda, rho, alpha, x_ini, x_g)

% lasso_lsqr Solve lasso problem via ADMM
%
% [z, history] = lasso_lsqr(A, b, lambda, rho, alpha);
%
% Solves the following problem via ADMM:
%
%   minimize 1/2*|| Ax - b ||_2^2 + \lambda || x ||_1,
%
% where A is a sparse matrix. This uses LSQR for the x-update instead.
%
% The solution is returned in the vector x.
%
% history is a structure that contains the objective value, the primal and
% dual residual norms, and the tolerances for the primal and dual residual
% norms at each iteration.
%
% rho is the augmented Lagrangian parameter.
%
% alpha is the over-relaxation parameter (typical values for alpha are
% between 1.0 and 1.8).
%
%
% More information can be found in the paper linked at:
% http://www.stanford.edu/~boyd/papers/distr_opt_stat_learning_admm.html
%

t_start = tic;


QUIET    = 0;
MAX_ITER = 1000;
ABSTOL   = 1e-10;
RELTOL   = 1e-10;


[m, n] = size(A);

x = zeros(n,1);
z = zeros(n,1);
u = zeros(n,1);


if ~QUIET
    fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
    'lsqr iters', 'r norm', 'eps pri', 's norm', 'eps dual', 'objective');
end
figure;
plot(x_g);
hold on;

for k = 1:MAX_ITER

    % x-update with lsqr; uses previous x to warm start
    [x, flag, relres, iters] = lsqr([A; sqrt(rho)*speye(n)], ...
        [b; sqrt(rho)*(z-u)], [], [], [], [], x);

    plot(x);
    hold on;
    pause(0.05);

%     if(flag ~=0)
%         error('LSQR problem...\n');
%     end

    % z-update with relaxation
    zold = z;
    x_hat = alpha*x + (1 - alpha)*zold;
    z = shrinkage(x_hat + u, lambda/rho);

    u = u + (x_hat - z);

    % diagnostics, reporting, termination checks
    history.objval(k)  = objective(A, b, lambda, x, z);

    history.lsqr_iters(k) = iters;
    history.r_norm(k)  = norm(x - z);
    history.s_norm(k)  = norm(-rho*(z - zold));

    history.eps_pri(k) = sqrt(n)*ABSTOL + RELTOL*max(norm(x), norm(-z));
    history.eps_dual(k)= sqrt(n)*ABSTOL + RELTOL*norm(rho*u);

    if ~QUIET
        fprintf('%3d\t%10d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', k, ...
            sum(history.lsqr_iters), history.r_norm(k), history.eps_pri(k), ...
            history.s_norm(k), history.eps_dual(k), history.objval(k));
    end

    if (history.r_norm(k) < history.eps_pri(k) && ...
      history.s_norm(k) < history.eps_dual(k))
        break;
    end

end

if ~QUIET
    toc(t_start);
end

end

function p = objective(A, b, lambda, x, z)
    p = ( 1/2*sum((A*x - b).^2) + lambda*norm(z,2) );
end

function z = shrinkage(x, kappa)
    z = max( 0, x - kappa ) - max( 0, -x - kappa );
end

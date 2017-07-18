function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% Compute the cost and gradients
for i = 1:m
    % compute the added value to the cost function in this iteration
    iter = y(i) * log(sigmoid(theta'*X(i,:)')) + ...
        (1-y(i)) * log(1 - sigmoid(theta'*X(i,:)'));
    J = J + iter;
    for j = 1:size(theta)
        grad(j) = grad(j) + X(i,j)*(sigmoid(theta'*X(i,:)')-y(i));
    end
end
J = J / (-m);
grad = grad./m;

% Add regularization term  to cost function (loop from 2 to n)
regul = 0;
for j = 2:size(theta)
    regul = regul + theta(j)^2;
end
regul = regul * (lambda / (2*m));
J = J + regul;

% Add regularization terms to gradients
for j = 2:size(theta)
    grad(j) = grad(j) + (lambda / m) * theta(j);
end

% =============================================================

end

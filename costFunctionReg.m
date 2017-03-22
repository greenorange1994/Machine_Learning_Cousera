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
[m, n] = size(X)
z = X * theta;
g = sigmoid(z);
for i = 1:m
    J = J + log(g(i))*(-y(i))-(1-y(i))*log(1-g(i))
end
grad(1) = 1/m*(X(:,1))'*(g-y);
for j = 2:n
    J = J + lambda/2*theta(j)^2;
    grad(j) = 1/m*(X(:,j))'*(g-y) + lambda/m*theta(j);
end
J = J*1/m;
% =============================================================

end

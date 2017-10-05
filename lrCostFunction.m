function [J, grad] = lrCostFunction(theta, X, y, lambda)


m = length(y); 

J = 0;
grad = zeros(size(theta));



all_h = sigmoid(X * theta);
no_first_theta = theta(2:size(theta,1));

J = (1 / m) * (-y' * log(all_h) - (1 - y)' * log(1 - all_h)) + (lambda / (2 * m)) ...
* (no_first_theta' * no_first_theta);

grad = (1 / m) * X' * (all_h - y);

grad(2:size(grad,1)) = grad(2:size(grad,1)) + (lambda / m) * no_first_theta;



grad = grad(:);

end

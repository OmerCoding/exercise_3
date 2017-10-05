function p = predictOneVsAll(all_theta, X)

m = size(X, 1);
num_labels = size(all_theta, 1);

p = zeros(size(X, 1), 1);

X = [ones(m, 1) X];


results = (sigmoid(X * all_theta'))';

[val, index] = max(results);

p = index(:);




end

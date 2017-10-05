function p = predict(Theta1, Theta2, X)

m = size(X, 1);
num_labels = size(Theta2, 1);

p = zeros(size(X, 1), 1);


ones_to_X = ones(m, 1);
X = [ones_to_X X];

a_2_mat = sigmoid(Theta1 * X');
a_2_mat = [ones_to_X';a_2_mat];

a_3_mat = [Theta2 * a_2_mat];

[val, index] = max(a_3_mat);

p = index(:);




end

function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add a column of 1's to X to make it a1
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Compute the linear combinations of weights and inputs for layer 2
z2 = X * Theta1';
% Compute the activation of the units: use the sigmoid function
a2 = sigmoid(z2);
% a2, the input to the output layer, is z2 with the biases added
% a2 is an (m x 26) matrix: for each training example, the initial
% 400 features have been 'simplied' into 25, and biases added in front
a2 = [ones(m,1) a2];

% Compute the linear combinations of weights and inputs for layer 3
z3 = a2 * Theta2';
% Compute the activation of the units: use the sigmoid function
% a3 is now an (m x 10) matrix, each row representing the predictions
% for that training example
a3 = sigmoid(z3);

% Compute the predictions: find the max index along each row,
% which represents the class with the highest probability
[~, p] = max(a3, [], 2);








% =========================================================================


end

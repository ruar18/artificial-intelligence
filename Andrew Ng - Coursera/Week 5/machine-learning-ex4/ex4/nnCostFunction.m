function [J, grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

% Make a matrix of one-hot encoded vectors for output labels
% Make an eye matrix, with each row being one potential one hot encoding
eye_matrix = eye(num_labels);
% Make a y-matrix: for each value in y, go to the yth line in eye_matrix
% and copy it over as the yth row vector in y-matrix
y_matrix = eye_matrix(y, :);

% Add column of ones
a1 = [ones(m, 1) X];
% Calculate z(2)
z2 = a1 * Theta1';
% Calculate a(2), first saving a variable to use later in backprop
g2 = sigmoid(z2);

% Add column of ones
a2 = [ones(m, 1) g2];
% Calculate z(3)
z3 = a2 * Theta2';
% Calculate a(3) which are the predictions
a3 = sigmoid(z3);

% Compute the cost with a vectorized implementation. Note that we use
% element-wise multiplication, since matrix multiplication is not so
% useful here. We need to sum along 2 dimensions to compensate for the 
% lack of inefficient for loops. First, sum along the rows, then columns.
J = sum(sum(y_matrix .* log(a3) + (1-y_matrix) .* log(1-a3),2));
J = J / (-m);

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

% Directly compare our outputs to the to the labels to compute the errors
% for layer 3
d3 = a3 - y_matrix;

% Compute deltas for the hidden layer. g'(z) is g(z)(1-g(z)). In this case,
% we have already computed g(z) = g2.
d2 = (d3 * Theta2);
% Exclude biases
d2 = d2(:, 2:end);
d2 = d2 .* (g2 .* (1-g2));

% Calculate the "accumulators", i.e. error for units based on all training
% examples
delta1 = d2' * a1; 
delta2 = d3'* a2;

% Calculate the Jacobian for each layer by dividing by m (taking the average)
Theta1_grad = delta1 / m;
Theta2_grad = delta2 / m;


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2

% Compute the double sum of all weights squared (excluding bias)
Theta1_sum = sum(sum(Theta1(:, 2:end) .^ 2, 2));
Theta2_sum = sum(sum(Theta2(:, 2:end) .^ 2, 2));

% Calculate regularization term
regular_term = (Theta1_sum + Theta2_sum) * (lambda / (2 * m));
% Add regularization terms to cost
J = J + regular_term;

% Add regularization to the gradient
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + (Theta1(:, 2:end) * ...
    (lambda / m));
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + (Theta2(:, 2:end) * ...
    (lambda / m));

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

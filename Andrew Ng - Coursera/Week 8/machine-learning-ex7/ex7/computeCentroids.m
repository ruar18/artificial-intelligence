function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

% for i = 1:K
%     centroids(i, :) = mean(X(idx == i, :));
% end

% We could use a for loop, but where's the fun in that. This is an obscure
% and extremely efficient vectorized implementation of the same thing
% accomplished above in the commented code. Using the example data in the
% assignment, we get a 3x speedup using this approach.

% Convert the centroid indices into binary matrix, c_mat is k by m
eye_mat = eye(K);
c_mat = eye_mat(idx, :)';
% Multiply by X to find sums of features of each example that is assigned
% to a centroid (i.e. this is the step before dividing by the number of
% points assigned to the centroid; we're almost done computing the mean)
centroids = c_mat * X;
% Summing c_mat row-wise gives us the number of points assigned to the
% centroid that corresponds to each row. We've successfully computed the
% mean.
centroids = centroids ./ sum(c_mat, 2);




% =============================================================


end


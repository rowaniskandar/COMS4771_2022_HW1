%COMS4771 summer B 2022
%homework 1
%codes for cross-validation are based on: https://mccormickml.com/2013/08/01/k-fold-cross-validation-with-matlab-code/
%created by: Rowan Iskandar (ri2282@columbia.edu;
%rowan.iskandar@sitem-insel.ch)
%modified: 16 July 2022
rng(4)
load("digits.mat")
features = zeros (10000,784);
%k-fold cross-validation performance evaluation
%number of k
k_cv = 10;
% List digits
categories = [0; 1; 2; 3; 4; 5; 6; 7; 8; 9];
% Get the number of vectors belonging to each category.
vecsPerCat = getVecsPerCat(X, Y, categories);
% Compute the fold sizes for each category
foldSizes = computeFoldSizes(vecsPerCat, k_cv);
% Randomly sort the vectors in X, then organize them by category.
[X_sorted, y_sorted] = randSortAndGroup(X, Y, categories);
performance_knn=zeros(k_cv,1);
for i=1:k_cv
    %iterate through each fold of cross-validation samples
    [X_train, Y_train, X_test, Y_test] = getFoldVectors(X_sorted, y_sorted, categories, vecsPerCat, foldSizes, i);
    %k-nn classifier
    num_k = 10;
    performance_knn(i) = classifier_knn_err(X_train,Y_train,X_test, Y_test,num_k);
end
%summary of performance
performance_knn_avg = sum(performance_knn)/k_cv;

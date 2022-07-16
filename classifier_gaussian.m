function class = classifier_gaussian(x_test,mean_hat,var_hat)
    score = zeros(10);
    for k=1:10
        score(k)= mvnpdf(x_test.',mean_hat(:,k),var_hat(:,:,k))*prior(k);
    end
    score_optim = max(score);
    class = find(score==score_optim)-1;
end
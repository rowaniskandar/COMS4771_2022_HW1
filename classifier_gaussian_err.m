function error = classifier_gaussian_err(x_test,y_test, mean_hat,var_hat)
    k_miss=0;
    for i=1:size(x_test,1)
        k_pred=classifier_gaussian(x_test(i,:),mean_hat,var_hat);
        if k_pred==y_test(i)
            k_miss=k_miss;
        else
            k_miss=k_miss+1;
        end
    end
    error=k_miss/size(x_test,1);
end
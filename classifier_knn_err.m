function error = classifier_knn_err(x_test,y_test,X_train, Y_train, k)
    k_miss=0;
    for i=1:size(x_test,1)
        k_pred=maxkNN(k,X_train, Y_train, x_test(i,:));
        %disp(k_pred);
        if k_pred==y_test(i)
            k_miss=k_miss;
        else
            k_miss=k_miss+1;
        end
    end
    error=k_miss/size(x_test,1);
    disp(error);
end
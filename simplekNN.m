function p = simplekNN(k, A, C, t)
    % compute Euclidian distance between test point and train points
    r_zx = sum(bsxfun(@minus, A, t).^2, 2);
    % sort the distances in ascending order
    [r_zx,idx] = sort(r_zx, 1, 'ascend');
    % keep only the K nearest neighbours
    r_zx = r_zx(1:k); % keep the first ’K’ distances
    idx = idx(1:k); % keep the first ’K’ indexes
    % majority vote 
    p = mode(C(idx));
end

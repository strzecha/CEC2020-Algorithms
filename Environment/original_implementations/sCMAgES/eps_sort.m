%% Function that sorts the (fitness,constraint violation)-pairs
%  specified in the vectors fit and cvio (according to the epslion level ordering)
function [ranking]=eps_sort(fit,cvio,epsilon)
    n = length(fit);
    ind   = linspace(1,n,n);
    for i=n-1:-1:1
        for j=1:i
            if eps_rank(fit(j),cvio(j),fit(j+1),cvio(j+1),epsilon)==0
                k=ind(j);
                f=fit(j);
                c=cvio(j);
                ind(j)=ind(j+1);
                fit(j)=fit(j+1);
                cvio(j)=cvio(j+1);
                ind(j+1)=k;
                fit(j+1)=f;
                cvio(j+1)=c;
            end
        end
    end
    ranking    = ind;       % return a vector of sorted indices
end
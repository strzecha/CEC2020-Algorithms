%% Function that ranks two (objective fucntion, constraint violation)-pairs 
%  according to a given epsilon-threshold 
%  OUTPUT: 1 -- if (f1,cv1) <_eps (f2,cv2)
%          0 -- else
%%
function [z]=eps_rank(f1,cv1,f2,cv2,epsilon)
    if cv1 == cv2
        z=(f1 <= f2);  
    elseif (cv1 <= epsilon) && (cv2<= epsilon)
        z=(f1 <= f2);
    else
        z=(cv1 < cv2);
    end
end
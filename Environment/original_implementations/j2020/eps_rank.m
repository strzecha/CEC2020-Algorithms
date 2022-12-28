function [z]=eps_rank(f1,cv1,f2,cv2,epsilon)
    if cv1 == cv2
        z=(f1 <= f2);  
    elseif (cv1 <= epsilon) && (cv2<= epsilon)
        z=(f1 <= f2);
    else
        z=(cv1 < cv2);
    end
end
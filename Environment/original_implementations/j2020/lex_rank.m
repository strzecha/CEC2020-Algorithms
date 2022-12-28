function [z]=lex_rank(f1,cv1,f2,cv2)
    if (cv1 == 0 && cv2 == 0) 
        z=(f1 <= f2);  
    else
        z=(cv1 <= cv2);
    end
end

function [vio_det]=violation_2(g,h, DELTAinq, gn, hn, DELTA,chk_opt )
vio_det=zeros(1,gn+hn);
if chk_opt==1
    DELTAinq= DELTAinq.*0.0;
    DELTA= 0.0001.* (ones(1,hn));
end
if gn>0
    vio_det(1,1:gn)= max(0,g-DELTAinq);
end

if hn>0
    for j = 1: hn
        if abs(h(j))>DELTA(j)
            vio_det(1,gn+j)= abs(h(j));
        else
            vio_det(1,gn+j)= 0.0;
        end
    end
end

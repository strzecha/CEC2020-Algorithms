function [Par] = Update_delta (Par,I_fno,current_eval)

if Par.hn>0
    
    Par.DELTA=ones(1,Par.hn)*0.0001;
    
end
if Par.gn>0
    Par.DELTAinq=zeros(1,Par.gn);
end


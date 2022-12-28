function [r1, r2,r3,r4] =gn_rand(PopSize, NP2)
r1=zeros(1,PopSize);
r2=zeros(1,PopSize);
r3=zeros(1,PopSize);
r4=zeros(1,PopSize);

% r4=zeros(1,PopSize);
% 
% r5=zeros(1,PopSize);
% r6=zeros(1,PopSize);
% r7=zeros(1,PopSize);

for i=1:PopSize
    r1(i)=ceil(rand*PopSize);
    while  r1(i)==i
        r1(i)=ceil(rand*PopSize);
    end
    r2(i)=ceil(rand*PopSize);
    while  r2(i)==r1(i) || r2(i)==i
        r2(i)=ceil(rand*PopSize);
    end
    r3(i)=ceil(rand*NP2);
    while r3(i)==r2(i) ||r3(i)==r1(i)|| r3(i)==i
        r3(i)=ceil(rand*NP2);
    end
    
    r4(i)=ceil(rand*PopSize);
    while r4(i)==r3(i) ||r4(i)==r2(i) ||r4(i)==r1(i)|| r3(i)==i
        r4(i)=ceil(rand*PopSize);
    end
%     
%     r5(i)=ceil(rand*PopSize);
%     while r5(i)==r4(i) || r5(i)==r3(i) ||r5(i)==r2(i) ||r5(i)==r1(i)|| r3(i)==i
%         r5(i)=ceil(rand*PopSize);
%     end
%     
%     
%     r6(i)=ceil(rand*PopSize);
%     while r6(i)==r5(i) ||r6(i)==r4(i) ||r6(i)==r3(i) ||r6(i)==r2(i) ||r6(i)==r1(i)|| r3(i)==i
%         r6(i)=ceil(rand*PopSize);
%     end
%     
%     r7(i)=ceil(rand*PopSize);
%     while r7(i)==r6(i) ||r7(i)==r5(i) ||r7(i)==r4(i)||r7(i)==r3(i) ||r7(i)==r2(i) ||r7(i)==r1(i)|| r3(i)==i
%         r7(i)=ceil(rand*PopSize);
%     end
    
    
end
end
clc;
clear;

for j = 1:57
if j < 10
      exp1 = ['COLSHADE_RC0' num2str(j) '_F.txt'];
      exp2 = ['COLSHADE_RC0' num2str(j) '_CV.txt'];
else
      exp1 = ['COLSHADE_RC' num2str(j) '_F.txt'];
      exp2 = ['COLSHADE_RC' num2str(j) '_CV.txt'];
end
      OB = load(exp1);
      CONV = load(exp2);
      F(:,j) = OB(10,:)';
      C(:,j) = CONV(10,:)';    
      FF = F(:,j)+1e37*C(:,j);
      [~,k] = sort(FF);
      F(:,j) = F(k,j);
      C(:,j) = C(k,j);
end
xlswrite('\COLSHADE.xlsx',F,1);
xlswrite('\COLSHADE.xlsx',C,2);

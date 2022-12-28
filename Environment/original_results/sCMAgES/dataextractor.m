clc;
clear;

for j = 1:57
      exp = ['sCMAgES' num2str(j) '.mat'];
      load(exp);
%       exp1 = ['sCMAgES' num2str(j) '_F.txt'];
%       exp2 = ['sCMAgES' num2str(j) '_CV.txt'];
%       OB = load(exp1);
%       CONV = load(exp2);
      F(:,j) = OB(10,:)';
      C(:,j) = CONV(10,:)';  
      FF = F(:,j)+1e37*C(:,j);
      [~,k] = sort(FF);
      F(:,j) = F(k,j);
      C(:,j) = C(k,j);
end 
xlswrite('\sCMAgES.xlsx',F,1);
xlswrite('\sCMAgES.xlsx',C,2);

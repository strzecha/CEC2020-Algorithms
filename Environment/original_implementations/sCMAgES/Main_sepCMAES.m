%%-------------------------------------------------------------------------------------------------------------------%%
% Authors: Abhishek Kumar, Swagatam Das, Ivan Zelinka
% Title: A Modified Covariance Matrix Adaptation Evolution Strategy for Real-World Constrained Optimization Problems 
%%-------------------------------------------------------------------------------------------------------------------%%
clear all;
clc
warning('off','all');
global  initial_flag initial_flag2    
initial_flag = 0;
initial_flag2 = 0;
foldername = 'Result_Record';
efn = exist(foldername);
if efn ~= 7
   mkdir(foldername);
end
for k = 21

   par = Cal_par(k);
   D   = par.n;
   problem.gn = par.g;
   problem.hn = par.h;
   if D <= 10
       MaxFES = 1e5;
   elseif D <= 30 ||D > 10
       MaxFES = 2e5;
   elseif D <= 50 ||D > 30
       MaxFES = 4e5;
   elseif D <= 150||D > 50
       MaxFES = 8e5;
   else
       MaxFES = 1e6;
   end
   input.lb = par.xmin;
   input.ub = par.xmax;
        %% input -- initial (fixed) strategy parameter setting
        input.budget            = MaxFES;
        input.delta             = 10^-4;                    
        input.runs              = 25;                     
        input.dim               = D;


        %% CMA specific paprameters      
        input.lambda  = 4+floor(3*log(D));           
        input.sigma   = 1;                            
        input.mu      = floor(input.lambda/3);       
        input.weights = log(input.mu+1/2)-log(1:input.mu)';     
        input.weights = input.weights./sum(input.weights);    
        input.mueff   = 1/sum(input.weights.^2);                    
        input.cs      = (input.mueff+2) / (D+input.mueff+3);         
        input.damps   = 1 + 2*max(0, sqrt((input.mueff-1)/(D+1))-1) + input.cs;                      
        input.cmu     = ( ( D + 2 ) / 3 ) * ( ( 2 / ( input.mueff * ( D + sqrt(2) ) ^ 2 ) ) + ( 1 - 1 / input.mueff)...
                        * min( [1, (2*input.mueff-1) / ((D+2)^2+input.mueff) ] ));
        input.cc      = 4/(D+4);  
        %% problem specification
        problem.constr_fun_name = 'cec20_func';                 
        strategy                = 'sepCMAESr';  
        %% Display
        disp(['ES variant ' strategy ' --- 25 independent runs!'])
        filename      = 'Result_Record\sCMAgESr_temp.xlsx';                        
        func_num      = k;                                      
        initial_flag  = 0;
        initial_flag2 = 0;
        %% Problem Iteration
            problem.upper_bounds    = input.ub';
            problem.lower_bounds    = input.lb';
            for j=1:input.runs                                 
                [tab,global_best, dta]= sepCMAESr(problem,input,func_num);
                FitT(j,:)=[tab(1,:) tab(2,:) tab(3,:)];
                OB(:,j) = dta(:,1);
                CONV(:,j) = dta(:,2);
                disp([func_num, j, OB(10,j), CONV(10,j)]);
            end
        %% Statistical Data
            Stats = xlsread(filename,1); Stats = Stats';
            Tab = build_stats(FitT,input.runs);                         
            Stats(k,:) = [func_num Tab];
            xlswrite(filename,Stats',1);
            filename1 = strcat(['Result_Record\sCMAgES' num2str(k) '_F' '.txt']);
            filename2 = strcat(['Result_Record\sCMAgES' num2str(k) '_CV' '.txt']);
            filename3 = strcat(['Result_Record\sCMAgES' num2str(k) '.mat']);
            save(filename1,'OB','-ascii');
            save(filename2,'CONV','-ascii');
            save(filename3,'OB','CONV');
            disp([func_num Tab]);
end
              

        


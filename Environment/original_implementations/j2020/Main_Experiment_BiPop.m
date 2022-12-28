% Main routine for running the BiPop-epsilonMAg-ES on the 2020 CEC constrained real-world benchmarks
clear all, clc

% definition of global variables for logging the required algortihm
% statistics in accordance with the competition guidelines.
global initial_flag
global logF
global logC
global logEvals
global fcflag;

for i = 1:57
   
    initial_flag = 0;
    % determine CEC problem specific parameters
        par = Cal_par(i);

    % choose the problem dimensionality D  
        D = par.n;

    % bound constraint definitions for the test functions under consideration
        Xmin = par.xmin;
        Xmax = par.xmax;

    % number of constraint functions of current problem
        problem.gn = par.g;
        problem.hn = par.h;

    % budget of function evaluations and generations depending on dimension D
        if D<=10
            MaxFES = 1*10^5;
        elseif D>10 && D<=30
            MaxFES = 2*10^5;
        elseif D>30 && D<=50
            MaxFES = 4*10^5;
        elseif D>50 && D<=150
            MaxFES = 8*10^5;
        else 
            MaxFES = 1*10^6;
        end
        
        func_num=i;       % number of the constrained test function to be evaluated
            
        eval(['problem.lower_bounds=transpose(Xmin);']);
        eval(['problem.upper_bounds=transpose(Xmax);' ]); 
        
        problem.constr_fun_name = 'cec20_func';  % objective function class

     %% Input parameter definitions and strategy parameters of the Evolution Strategy
     %
        input.dim               = D;
        input.budget            = MaxFES;
        input.delta             = 10^-4;                    % error margin for equality constraints
        input.runs              = 25;                       % number of independent algorithm runs

        input.nu                = 1/3;    
        input.lambda            = (4+ floor(3*log(D)));           % population size
        input.mu                = ceil(input.lambda*input.nu);    % parental ppopulation size

        input.sigma             = 1;                        % initial mutation strength
        input.T                 = 500;                      % maximum number of generations to perform the epsilon-level constrained handling
        
        


    % MA-ES specific strategy paprameters (standard parameter choices)
    
        input.weights = log(input.mu+1/2)-log(1:input.mu)';     % mu \times one array for weighted recombination
        input.weights = input.weights./sum(input.weights);      % normalize recombination weights array
        input.mueff=1/sum(input.weights.^2);                    % variance-effectiveness of sum w_i x_i

        input.cs = (input.mueff+2) / (D+input.mueff+5);         % time - const for cumulation for sigma control
        input.c1 = 2 / ((D+1.3)^2+input.mueff);                 % learning rate for rank-one update of M
        input.cmu = min(1-input.c1, 2 * (input.mueff-2+1/input.mueff) / ((D+2)^2+input.mueff));      % and for rank-mu update of M
        input.damps = 1 + 2*max(0, sqrt((input.mueff-1)/(D+1))-1) + input.cs;                       % damping for sigma usually close to 1

    % BiPop-MA-ES specific strategy parameters    
        input.thetap    = 0.2; % probability to repair a candidate solution
        input.reps      = 3;   % number of repair repetitions within epsMAg-ES
        input.cp        = 2;   % parameter for epsilon-level reduction
        

    % MA-ES variant for constrained optimization
        strategy='BiPopEpsMAgES';

        foldername = ['CEC2020_WhiteBox_CompetitionResults_' strategy]; 
        efn = exist(foldername);
        if efn ~= 7
            mkdir(foldername);
        end
        i
    % on each of the selected CEC2017 test function do         
        for j=1:input.runs  
            logF =zeros(1,9);
            logC =zeros(1,9); 
            logEvals = zeros(1,9);
            fcflag = 1;
            
            eval(['[global_best]=' strategy '(problem,input,func_num);']); % run strategy

            GB{j}           = global_best;                  % store globally best observation of run j
        
            Tf(:,j)= [logF, global_best.val];
            Tc(:,j)= [logC, global_best.conv];
            Te(:,j)= [logEvals, input.budget];
        end
            
        save([foldername '/' strategy '_RS_on_fun' num2str(func_num) 'D' num2str(D) '.mat'],'GB','problem','input','Tf','Tc','Te','-v7')
        
        Tf=[];
        Tc=[];
        Te=[];
end

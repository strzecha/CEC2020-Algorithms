% The outer loop of the BiPop-epsilonMAg-ES 
function [global_best]=BiPopEpsMAgES(problem,input,CEC_fun_no)
    
    global fcflag 
    global logF 
    global logC

    yPopInit = problem.lower_bounds+rand(input.dim,input.lambda).*(problem.upper_bounds-problem.lower_bounds);    
    global_old = [];
    
    % performe 1st epsMAgES run
    [gb,evals] = epsMAgESbp_wb(problem,input,CEC_fun_no,0,yPopInit,global_old);
    global_best = gb;
    global_old  = gb;
    NoEvals     = evals.fun;
    
    % set parameters for counting and controlling population sizes in
    % restarts
    n  = 0;
    ns = 0;
    bl = 0;
    bs = 0;
        
    while NoEvals<=input.budget
        n = n+1;
        
        % population control
        lambda = 2^(n-ns)*input.lambda;
        
        % variation of strategy parameters
        if global_best.conv > 0 && mod(n,2)==1
            input.reps = 20;
            input.T = 0;
        else
            input.reps = 3;
            input.T = 500;
        end
            
            
        
        if n > 2 && bs<bl
            % CASE 1: initialize the use small populations
            sigma           = input.sigma; %/ 100^(rand);
            lams            = floor(input.lambda*(lambda/2/input.lambda)^((rand)^2));
            yPopInit        = problem.lower_bounds+rand(input.dim,lams).*(problem.upper_bounds-problem.lower_bounds); 
            
            input3          = input;
            input3.lambda   = lams;
            input3.sigma    = sigma;
            input3.mu       = ceil(input3.lambda*input.nu);
            input3.weights  = log(input3.mu+1/2)-log(1:input3.mu)';         % mu \times one array for weighted recombination
            input3.weights  = input3.weights./sum(input3.weights);          % normalize recombination weights array
            input3.mueff    = 1/sum(input3.weights.^2);                     % variance-effectiveness of sum w_i x_i
                        
            input3.cs = (input3.mueff+2) / (input.dim+input3.mueff+5);      % time - const for cumulation for sigma control
            input3.c1 = 2 / ((input.dim+1.3)^2+input3.mueff);               % learning rate for rank-one update of M
            
            input3.cmu      = min(1-input3.c1, 2 * (input3.mueff-2+1/input3.mueff) / ((input.dim+2)^2+input3.mueff));   % and for rank-mu update of M
            input3.damps    = 1 + 2*max(0, sqrt((input3.mueff-1)/(input.dim+1))-1) + input3.cs;                       % damping for sigma usually close to 1
            
            % run epsilonMAgES 
            [gb, evals] = epsMAgESbp_wb(problem,input3,CEC_fun_no,NoEvals,yPopInit,global_old);
            
            bs      = bs + (evals.fun-NoEvals);
            NoEvals = evals.fun;
            ns      = ns + 1;
            
            % update global best solution
            if lex_rank(global_best.val,global_best.conv,gb.val,gb.conv)==0
                global_best=gb;
            end
            
        else 
            % CASE 1: initialize the use larger populations
            input2 = input;
            input2.lambda = lambda;
            yPopInit = problem.lower_bounds+rand(input.dim,lambda).*(problem.upper_bounds-problem.lower_bounds); 
            
            input2.mu                = ceil(input2.lambda*input.nu);
            input2.weights = log(input2.mu+1/2)-log(1:input2.mu)';     % mu \times one array for weighted recombination
            input2.weights = input2.weights./sum(input2.weights);      % normalize recombination weights array
            input2.mueff=1/sum(input2.weights.^2);                    % variance-effectiveness of sum w_i x_i

            input2.cs = (input2.mueff+2) / (input.dim+input2.mueff+5);         % time - const for cumulation for sigma control
            input2.c1 = 2 / ((input.dim+1.3)^2+input2.mueff);                 % learning rate for rank-one update of M
            input2.cmu = min(1-input2.c1, 2 * (input2.mueff-2+1/input2.mueff) / ((input.dim+2)^2+input2.mueff));      % and for rank-mu update of M
            input2.damps = 1 + 2*max(0, sqrt((input2.mueff-1)/(input.dim+1))-1) + input2.cs;                       % damping for sigma usually close to 1
           
            % run epsilonMAgES 
            [gb, evals] = epsMAgESbp_wb(problem,input2,CEC_fun_no,NoEvals,yPopInit,global_old);
            
            bl = bl + (evals.fun-NoEvals);
            NoEvals = evals.fun;

            % update global best solution
            if lex_rank(global_best.val,global_best.conv,gb.val,gb.conv)==0
                global_best=gb;
            end
            
        end
        
        global_old  = global_best;
    end
    

end


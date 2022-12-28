function [global_best, evals]=epsMAgESbp_wb(problem,input,CEC_fun_no,fevals,yPopInit,global_old)
    %% Implementation of the epsilonMAg-ES for constrained optimiazation
    % Implementation according to
    % M. Hellwig and H.-G. Beyer, "A Matrix Adaptation Evolution Strategy for Constrained
    % Real-Parameter Optimization", 2018 IEEE Congress on Evolutionary
    % Computation (CEC), IEEE, 2018, https://dx.doi.org/10.1109/CEC.2018.8477950
    %
    % with modifications for the use within the BiPop-epsilonMAg-ES
    % according to 
    % M. Hellwig and H.-G. Beyer, "A Modified Matrix Adaptation Evolution Strategy with 
    % Restarts for Constrained Real-World Problems", 2020 IEEE Congress on Evolutionary
    % Computation (CEC), IEEE, 2020, accepted.
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%% White-box version: The constraints are considered
    %%%%%%%%%%%%%%%%%%%% white-box. That is, the gradient repair assumes
    %%%%%%%%%%%%%%%%%%%% only one additional function evaluation per
    %%%%%%%%%%%%%%%%%%%% Jacobian approaximation  %%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % define global parameters for performance statistic measurements
    global fcflag 
    global logF 
    global logC
    
    % redefine some parameters
    dim         = input.dim;
    sigma       = input.sigma;
    mu          = input.mu;
    lambda      = input.lambda;
    newpop.y    = zeros(dim,lambda);					% initialize new population matrix (n times NP)
	newpop.f    = 0;
	newpop.conv = 0;
	evals.fun   = fevals;
	evals.con   = 0;
    g           = 0;
    termination = 0;
       
    % Initialize dynamic (internal) strategy parameters and constants
    ps      = zeros(dim,1);                                 % evolution paths for sigma
    M       = eye(dim);                                     % transformation matrix
    sqrt_s  = sqrt(input.cs*(2-input.cs)*input.mueff);      % factor in path update
    
    % upper limit of admissilbe mutation strength values
    sigmax = max(problem.upper_bounds-problem.lower_bounds)/2;     
    
    % Initialize random population of lambda individuals as done in the DE
    % variants
    for k=1:lambda
        % create initial population of uniformly distributed vectors within box-constraints 
            % newpop.y(:,k)   = problem.lower_bounds...
            % +(problem.upper_bounds-problem.lower_bounds).*rand(dim,1); 
        newpop.y(:,k)   = yPopInit(:,k);
        % newpop.y(:,k)   = keep_range(newy,problem.lower_bounds,problem.upper_bounds);
        % evaluate initial population
        [fval, gv, hv]  = feval(problem.constr_fun_name,newpop.y(:,k)',CEC_fun_no);
        newpop.f(k)     = fval;	  
        % calculate constraint violation according to CEC2017
        % recommendations
        newpop.conv(k)  = sum([sum(gv.*(gv>0)), sum(abs(hv).*(abs(hv)>input.delta))])./(problem.gn + problem.hn);
        % count constraint function evaluations
		evals.fun       = evals.fun + 1;
    end
       
    % initial parameters of the epsilon constraint handling approach
    TC=input.T;                                             
    EPSILON= median(newpop.conv);
    if input.T >0
        Epsilon= EPSILON;    
    else
        Epsilon = 0;
    end
    CP = input.cp; 
    
    [ranking]           = eps_sort(newpop.f,newpop.conv,Epsilon);                          % epsilon Ranking
    ParentPop           = newpop.y(:,ranking(1:mu));
    yParent             = sum(ParentPop,2)./mu;
    
    best_ind            = ranking(1);
    best_val            = newpop.f(best_ind);				% best fitness value of current population
    best_y              = newpop.y(:,best_ind);
    best_conv           = newpop.conv(:,best_ind);
        
    evals.sum           = evals.fun;
    
    % best solution found so far
    global_best.y       = best_y; 				
    global_best.val     = best_val;
    global_best.conv    = best_conv;
    global_best.evals   = evals.fun;
    global_best.logF 	= [0,0];
    global_best.logC 	= [0,0];

    logdata(global_best,evals.fun,input,global_old);
      
    while ~termination
        
        % compute the inverse of the transformation matrix for the
        % back-calculation approach
         Minv =  pinv(M,10^-12);

        % create new generation of offspring candidate solutions
        newpop.z    = randn(dim,lambda);
        newpop.d    = M*newpop.z;
        newpop.y    = repmat(yParent,1,lambda)+sigma.*newpop.d;
               
        for k=1:lambda          
            % initialization of repair count
            repi = 0;
            % check for bound constraint satisfaction and repair if
            % necessary
            newy = keep_range(newpop.y(:,k),problem.lower_bounds,problem.upper_bounds);
            % check whether repair has been carried out in order to apply
            % back-calculation,
            if ~isequal(newy,newpop.y(:,k))
                repi = 1;
            end
            
            % evaluation of offspring candiadate solution (in bounds)
            [fval, gv, hv]  = feval(problem.constr_fun_name,newy',CEC_fun_no);
            fitval          = fval;                                                                
            % compute constraint violation
            convio          = sum([sum(gv.*(gv>0)), sum(abs(hv).*(abs(hv)>input.delta))])./(problem.gn + problem.hn);            
            evals.fun       = evals.fun + 1;
            % initialize individual repair step count
            h=1;            
            % apply gradient-based mutation step if conditions are satisfied
            if mod(g,dim)==0 && rand(1) <= input.thetap %thetap = 0.2
                while convio > 0 && h <= input.reps
                     new_mutant = gradientMutation(problem,newy,gv,hv,CEC_fun_no);
                     new_mutant=keep_range(new_mutant,problem.lower_bounds,problem.upper_bounds);
            
                     [fval, gv, hv]  = feval(problem.constr_fun_name,new_mutant',CEC_fun_no);
                     fitval = fval;                                                               
                     convio = sum([sum(gv.*(gv>0)), sum(abs(hv).*(abs(hv)>input.delta))])./(problem.gn + problem.hn);
                     evals.fun       = evals.fun +1;                                    % white box version!!!
                     h=h+1;
                     newy = new_mutant;
                     if ~isequal(newy,newpop.y(:,k))
                        repi = 1;
                     end
                end 
                
            end
            newpop.y(:,k) = newy;
 
            % apply back-calculation if necessary
            if repi > 0
                newpop.d(:,k) = (newpop.y(:,k)-yParent)./sigma;
                newpop.z(:,k) = Minv*newpop.d(:,k);
            end
            newpop.f(k)     = fitval;                                                          
            newpop.conv(k)  = convio;
            

        end
           
        % Implementation of Epsilon Constraint Ordering 
        % feasible (constraint violation below epsilon value!!!) solutions dominate infeasible 
        % ones AND feasible solutions are sorted according to their fitness values
                
        [ranking]   = eps_sort(newpop.f,newpop.conv,Epsilon);
                
        best_ind    = ranking(1);
        best_val    = newpop.f(best_ind);             % best feasible fitness value of current population
        best_y      = newpop.y(:,best_ind);           % best feasible individual of current population
        best_conv   = newpop.conv(:,best_ind);
        
        % Sort by fitness and compute weighted mean into xmean
        parent_z = newpop.z(:,ranking(1:mu)) * input.weights;  % recombination
        parent_d = newpop.d(:,ranking(1:mu)) * input.weights;  % recombination
        yParent  = yParent + sigma *parent_d; % update population certroid
               
        % Cumulation: Update evolution paths
        ps = (1-input.cs) * ps + sqrt_s * parent_z; 

        % Update transformation matrix
            M = (1 - input.c1/2 - input.cmu/2).*M + input.c1./2.*(M*(ps*ps'));       
            for m = 1:mu
                M = M + (input.cmu/2*input.weights(m))*M*(newpop.z(:,ranking(m))*newpop.z(:,ranking(m))');
            end
            
        liMM = M > 1e+12 | isnan(M) | isinf(M);
        siMM = M < -1e+12 | isnan(M) | isinf(M);
        if sum(sum(liMM))>1 || sum(sum(siMM))>1         % reset transformation matirx if M deteriorates => pseudo inverse would collapse 
            M   = eye(input.dim);
            ps  = ones(dim,1);
        end    
            
        % Adapt mutation strength sigma
        sigma = min(sigma * exp((input.cs/2)*(norm(ps)^2/input.dim - 1)),sigmax);


        % update the best solution found so far                        
        if (best_conv==0 && global_best.conv==0 && best_val < global_best.val) ||...
                (best_conv==global_best.conv && best_val < global_best.val) || best_conv<global_best.conv
            global_best.y     = best_y; 				
            global_best.val   = best_val;
            global_best.conv  = best_conv;
            global_best.evals = evals.fun; % number of function evaluatioan at which best so far was observed
        end
	    logdata(global_best,evals.fun,input,global_old); % log performance statistics
        evals.sum = evals.fun;
                
        % termination criteria
        if evals.sum>=input.budget % budget exhaustion 
            termination = 1;
            disp('budget')
        elseif abs(evals.sum-global_best.evals)>input.budget*0.1 % stagation 
            termination = 1;
            disp('stagnation')
        elseif sigma < 10^-12   % too small mutation strength sigma
            termination = 1;
            disp('sigma')
        end
        
        % update generation counter
        g=g+1; 
        
        % ratio of epsilon-feasible candiate solutions in the population
        rat = sum(newpop.conv(ranking(1:mu))<=Epsilon)/mu; 

        % update epsilon-level threshold
        if(g>1 && g<TC) && rat >0.2 
            Epsilon=Epsilon*((1-g/TC)^CP);
        elseif (g>1 && g<TC) && rat <=0.2
            Epsilon=Epsilon*1.1;
        elseif(g+1>=TC)  || Epsilon < 10^-8
             Epsilon=0;
        end   
        
        
    end

    
end

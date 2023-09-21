function [out,global_best,dta] = sepCMAESr(problem,input,CEC_fun_no)
    %% Parameter Initialization
    dim               = length(problem.lower_bounds);
    sigma             = input.sigma;
    mu                = input.mu;
    lambda            = input.lambda;
    newpop.y          = zeros(dim,lambda);  
	newpop.f          = 0;
	newpop.conv       = 0;
	evals.fun         = 0;
    g                 = 0;
    termination       = 0;
    ps                = zeros(dim,1);                               
    pc                = zeros(dim,1);
    MM                = eye(dim);
    CC                = eye(dim);
    sqrt_s            = sqrt(input.cs*(2-input.cs)*input.mueff);      
    sqrt_c            = sqrt(input.cc*(2-input.cc)*input.mueff);  
    constraint_number = problem.gn + problem.hn;
    %% Initialize random population of lambda candidate solutions
    newpop.y        = problem.lower_bounds...
                            +(problem.upper_bounds-problem.lower_bounds).*rand(dim,lambda); 
    [fval, gv, hv]  = feval(problem.constr_fun_name,newpop.y',CEC_fun_no);
    newpop.f        = fval';                              
    newpop.conv     = (sum([sum(gv.*(gv>0),1); sum(abs(hv).*(abs(hv)>input.delta),1)],1)./constraint_number);  
    newpop.g        = gv;
    newpop.h        = hv;
    evals.fun       = evals.fun + lambda;
    %% Initial parameter for epsilon level ordering
    [TC,Eg,Eh,CPg,CPh] = ETAintialization(newpop);
    Eg0                = Eg;
    Eh0                = Eh;                   
    %% Rank initial population 
    [ranking]           = eta_sort(newpop,Eg,Eh);
    ParentPop           = newpop.y(:,ranking(1:mu));
    yParent             = sum(ParentPop,2)./mu;  
    %% Best individual of current population
    best_ind            = ranking(1);
    best_val            = newpop.f(best_ind);       
    best_y              = newpop.y(:,best_ind);
    best_conv           = newpop.conv(:,best_ind);         
    %% Best solution found so far
    global_best.y       = best_y; 			
	global_best.val     = best_val;
    global_best.conv    = best_conv;
    %% local intialization 
    local_best.y       = best_y; 			
	local_best.val     = best_val;
    local_best.conv    = best_conv;
    tolY               = 0;
    tolF               = inf;
    %% Upper mutation strength bound
    sigmaMAX            = 100;
    flag1               = 0;
    flag2               = 0;
    flag                = zeros(1,10);
    restart             = 0;
    %% Iteration Starts
    while ~termination
        %% Checking for the restart
        if restart == 1
            [sigma, g, ps, pc, MM, CC, FEs, Eg0, Eh0, yParent, TC, Eg, Eh, CPg, CPh, input ] = restarts(problem,input,CEC_fun_no);
            evals.fun       = evals.fun + FEs;
            restart         = 0;
            sqrt_s          = sqrt(input.cs*(2-input.cs)*input.mueff);    
            sqrt_c          = sqrt(input.cc*(2-input.cc)*input.mueff);
            lambda          = input.lambda;
            mu              = input.mu;
   %         tolY            = 0;
        end
        %% Compute inverse of transformation matrix MM
        piM       = diag(1./diag(MM));
        %% Sample lambda offspring distributed around yParent
        newpop.z        = randn(dim,lambda);
        newpop.d        = MM*newpop.z;
        newpopy         = yParent(:,ones(1,lambda)) + sigma.*newpop.d;
        newpop.y        = keep_range(newpopy,problem.lower_bounds,problem.upper_bounds,2);
        repi            = (sum(newpop.y~=newpopy,1) > 0);
        [fval, gv, hv]  = feval(problem.constr_fun_name,newpop.y',CEC_fun_no);
        evals.fun       = evals.fun + lambda;
        conv            = (sum([sum(gv.*(gv>0),1); sum(abs(hv).*(abs(hv)>input.delta),1)],1)./constraint_number);
        %% Solution Repair
        if (problem.hn > 0 && dim < 20) || problem.hn == 0
            problem.maxiter        = 4;
            pnb                    = 1;
        else
            problem.maxiter        = 3000;
            pnb                    = 0.2;
        end
        if mod(g,dim) == 0 
            for k = 1:lambda
                if rand <= pnb && conv(k) > 0 
                    problem.xmax                 = problem.upper_bounds;
                    problem.xmin                 = problem.lower_bounds;
                    problem.n                    = dim;
                    problem.I_fno                = CEC_fun_no;
                    [new_mutant,fes]              = QNrepair(problem,newpop.y(:,k),gv(:,k),hv(:,k));
                    new_mutant                   = keep_range(new_mutant,problem.lower_bounds,problem.upper_bounds,2);
                    [fval(k), gv(:,k), hv(:,k)]  = feval(problem.constr_fun_name,new_mutant',CEC_fun_no);                                                              % fitness vector 
                    conv(k)                      = sum([sum(gv(:,k).*(gv(:,k)>0)), sum(abs(hv(:,k)).*(abs(hv(:,k))>input.delta))])....
                                                   ./constraint_number;
                    evals.fun                    = evals.fun + fes + 1;     
                    newpop.y(:,k)                = new_mutant;
                    repi(k)                      = repi(k)+1;
                end
            end
        end
        %% modify datas of modified solution
        ll = find(repi > 0);
        newpop.d(:,ll) = (newpop.y(:,ll)    - yParent(:,ones(1,length(ll))))./sigma;
        newpop.z(:,ll) =  piM*newpop.d(:,ll);
        newpop.f       = fval';                                                                
        newpop.conv    = conv;
        newpop.g       = gv;
        newpop.h       = hv;
        ran            = eta_sort(newpop,zeros(size(Eg)),zeros(size(Eh)));
        [ranking]      = eta_sort(newpop,Eg,Eh);
        %% Best individual of current population
        best_ind    = ran(1);
        tolF        = abs(best_val-newpop.f(best_ind));
        best_val    = newpop.f(best_ind);            
        best_y      = newpop.y(:,best_ind);          
        best_conv   = newpop.conv(:,best_ind);      
        %% Recombination of mutation vectors and centroid update
        parent_z = newpop.z(:,ranking(1:mu)) * input.weights;  
        parent_d = newpop.d(:,ranking(1:mu)) * input.weights;
        yParent  = yParent + sigma * parent_d; 
        tolX     = max(abs(sigma * parent_d));                       
        %% Update evolution path and transformation matrix
        ps   = (1-input.cs) * ps + sqrt_s * parent_z; 
        hsig = sum(ps.^2)/(1-(1-input.cs)^(2*evals.fun/lambda))/dim < 2+4/(dim+1);
        pc   = (1-input.cc) * pc + hsig * sqrt_c * parent_d;        
        CC   = (1 - input.cmu) * CC + 1/input.mueff*(pc)*pc' + (input.cmu*(1-1/input.mueff)*newpop.d(:,ranking(1:mu)))...
               *diag(input.weights)*newpop.d(:,ranking(1:mu))';
        MM   = diag(diag(CC).^(0.5));
        %% Adapt the mutation strength 
        sigma = min(sigma  * exp((input.cs/2)*(norm(ps)^2/input.dim - 1)),sigmaMAX);
        %% check for degeneration
        degenerated = 0;
        if(max(sigma*diag(MM))<1e-128)
		   degenerated = 1;
        end
	    if(min(min(isfinite(CC))) == 0)
           degenerated = 1;
        else
            if(cond(MM)>1e16)
              degenerated = 1;
            end
		    if(~isreal(MM))
			  degenerated = 1;
            end
        end
	    if(degenerated)
		   CC = eye(dim);
		   MM = CC;
        end                 
        %% check for the termination
        if evals.fun>=input.budget                     
            termination = 1;
        end
        %% update local best solution found so far
        if (best_conv==0 && local_best.conv==0 && best_val <= local_best.val) ||...
                (best_conv==local_best.conv && best_val <= local_best.val) || best_conv<local_best.conv
            local_best.y    = best_y; 
            local_best.val  = best_val;
            local_best.conv = best_conv;
            local_best.fevs = evals.fun;
            tolY            = 0;
        else
            tolY            = tolY+1;
        end
        %% Repair of local best solution
        problem.xmax    = problem.upper_bounds;
        problem.xmin    = problem.lower_bounds;
        problem.n       = dim;
        problem.I_fno   = CEC_fun_no;
        problem.maxiter = 5000;
        if  local_best.conv ~= 0 && mod(g,dim) == 0 && problem.n >= 20 && problem.hn > 0 
          [~,gx,hx]        = cec20_func(local_best.y(:)',CEC_fun_no); 
          [new_mutant,fes] = QNrepair(problem,local_best.y(:),gx,hx);
          new_mutant       = keep_range(new_mutant,problem.lower_bounds,problem.upper_bounds,2);
          [fval, gv, hv]   = feval(problem.constr_fun_name,new_mutant',CEC_fun_no);                                                                
           conv            = sum([sum(gv.*(gv>0)), sum(abs(hv).*(abs(hv)>input.delta))])./constraint_number;
           evals.fun       = evals.fun + fes + 1;    
         if (conv==0 && local_best.conv==0 && fval <= local_best.val) ||...
                (conv==local_best.conv && fval <= local_best.val) || conv<local_best.conv
            local_best.y    = new_mutant; 
            local_best.val  = fval;
            local_best.conv = conv;
            local_best.fevs = evals.fun;
        end
        end
        %% update best solution found so far                    
        if (local_best.conv==0 && global_best.conv==0 && local_best.val <= global_best.val) ||...
                (local_best.conv==global_best.conv && local_best.val <= global_best.val) || local_best.conv<global_best.conv
            global_best.y    = local_best.y; 				
            global_best.val  = local_best.val;
            global_best.conv = local_best.conv;
            global_best.fevs = evals.fun;
        end  
        %% check for the restart
        if (tolX < 1e-25 || tolY > 300 || tolF < 1e-25) && g > TC
            restart         = 1;
            local_best.y    = []; 				
            local_best.val  = inf;
            local_best.conv = inf;
        end
        %% Update epsilon value 
        g    = g+1;  
        if(g>1 && g<TC)
          Eg = Eg0.*((1-g./TC).^CPg);
          Eh = Eh0.*(1-g./TC).^CPh;
        elseif(g+1>=TC)
          Eg = zeros(size(Eg));
          Eh = zeros(size(Eh));
        end 
      %% log global best after having used 10%, and 50% of the evaluation budget
        if evals.fun>=input.budget*10/100 && flag1==0
            fit10=global_best.val;
            con10=global_best.conv;
            [ff,gg,hh]=feval(problem.constr_fun_name,global_best.y',CEC_fun_no);
            c10_1    = sum(gg>1)                  + sum(abs(hh)>1);
            c10_2    = sum((gg>0.01) & (gg<1))    + sum(abs(hh)>0.01 & abs(hh)<1);
            c10_3    = sum((gg>0.0001)&(gg<0.01)) + sum(abs(hh)>0.0001 & abs(hh)<0.01);  
            flag1=1;
        elseif evals.fun>=input.budget*50/100 && flag2==0
            fit50=global_best.val;
            con50=global_best.conv;
            [ff,gg,hh]=feval(problem.constr_fun_name,global_best.y',CEC_fun_no);
            c50_1    = sum(gg>1)                  + sum(abs(hh)>1);
            c50_2    = sum((gg>0.01)&(gg<1))      + sum(abs(hh)>0.01 & abs(hh)<1);
            c50_3    = sum((gg>0.0001)&(gg<0.01)) + sum(abs(hh)>0.0001 & abs(hh)<0.01)  ;
            flag2=1;
        end
    %% calculation of Data
    if flag(1) == 0 && evals.fun >= 0.1*input.budget
       dta(1,:) = [global_best.val global_best.conv];
       flag(1) = 1;
    end
    if flag(2) == 0 && evals.fun >= 0.2*input.budget
       dta(2,:) = [global_best.val global_best.conv];
       flag(2) = 1; 
    end
    if flag(3) == 0 && evals.fun >= 0.3*input.budget
       dta(3,:) = [global_best.val global_best.conv];
       flag(3) = 1;
    end
    if flag(4) == 0 && evals.fun >= 0.4*input.budget
       dta(4,:) = [global_best.val global_best.conv];
       flag(4) = 1; 
    end
    if flag(5) == 0 && evals.fun >= 0.5*input.budget
       dta(5,:) = [global_best.val global_best.conv];
       flag(5) = 1;
    end
    if flag(6) == 0 && evals.fun >= 0.6*input.budget
       dta(6,:) = [global_best.val global_best.conv];
       flag(6) = 1;
    end
    if flag(7) == 0 && evals.fun >= 0.7*input.budget
       dta(7,:) = [global_best.val global_best.conv];
       flag(7) = 1; 
    end
    if flag(8) == 0 && evals.fun >= 0.8*input.budget
       dta(8,:) = [global_best.val global_best.conv];
       flag(8) = 1;
    end
    if flag(9) == 0 && evals.fun >= 0.9*input.budget
       dta(9,:) = [global_best.val global_best.conv];
       flag(9) = 1; 
    end
    if flag(10) == 0 && evals.fun >= 1*input.budget
       dta(10,:) = [global_best.val global_best.conv];
       flag(10) = 1;
    end

    end
    
    %% log final global best solution
    fit100=global_best.val;
    con100=global_best.conv;
    [ff,gg,hh]=feval(problem.constr_fun_name,global_best.y',CEC_fun_no);
    c100_1    = sum(gg>1)                   + sum(abs(hh)>1);
    c100_2    = sum((gg>0.01)&(gg<1))       + sum(abs(hh)>0.01 & abs(hh)<1);
    c100_3    = sum((gg>0.0001)&(gg<0.01))  + sum(abs(hh)>0.0001 &abs(hh)<0.01);

    out = [fit10 con10 c10_1 c10_2 c10_3;
             fit50 con50 c50_1 c50_2 c50_3;
             fit100 con100 c100_1 c100_2 c100_3];
    
end
function [sigma, g, ps, pc, MM, CC, FEs, Eg0, Eh0, yParent, TC, Eg, Eh, CPg, CPh, input ] = restarts(problem,input,CEC_fun_no)
    dim         = length(problem.lower_bounds);
    sigma       = input.sigma;
    if input.lambda < 8*dim
        if input.dim >= 25
           lambda    = floor(1.5*input.lambda);
        else
           lambda    = floor(1.2*input.lambda);
        end
    input.lambda = lambda;
    end
    mu              = floor(input.lambda/3);
    input.mu        = mu;  
    input.weights   = log(mu+1/2)-log(1:mu)';    
    input.weights   = input.weights./sum(input.weights);     
    input.mueff     = 1/sum(input.weights.^2);                   
    input.cs        = (input.mueff+2) / (dim+input.mueff+3);         
    input.damps     = 1 + 2*max(0, sqrt((input.mueff-1)/(dim+1))-1) + input.cs;                       
    input.cmu       = ((dim+2)/3)*((2/(input.mueff*(dim+sqrt(2))^2))+(1-1/input.mueff)...
                      *min([1,(2*input.mueff-1)/((dim+2)^2+input.mueff)]));
    input.cc        = 4/(dim+4);           
    newpop.y        = zeros(dim,input.lambda);   
	newpop.f        = 0;
	newpop.conv     = 0;
	FEs             = 0;
    g               = 0;   
    %% Initialize dynamic (internal) strategy parameters and constants
    ps                = zeros(dim,1);                                
    pc                = zeros(dim,1);
    MM                = eye(dim);
    CC                = eye(dim);   
    constraint_number = problem.gn + problem.hn;
    %% Initialize random population of lambda candidate solutions
    newpop.y        = problem.lower_bounds...
                            +(problem.upper_bounds-problem.lower_bounds).*rand(dim,input.lambda); 
    [fval, gv, hv]  = feval(problem.constr_fun_name,newpop.y',CEC_fun_no);
    newpop.f        = fval';                             
    newpop.conv     = (sum([sum(gv.*(gv>0),1); sum(abs(hv).*(abs(hv)>input.delta),1)],1)./constraint_number);  
    newpop.g        = gv;
    newpop.h        = hv;
    FEs             = FEs + input.lambda;
    %% Initial parameter for epsilon level ordering
    [TC,Eg,Eh,CPg,CPh] = ETAintialization(newpop);
    Eg0                = Eg;
    Eh0                = Eh;
    [ranking]          = eta_sort(newpop,Eg,Eh);
    ParentPop          = newpop.y(:,ranking(1:mu));
    yParent            = ParentPop*input.weights;
end
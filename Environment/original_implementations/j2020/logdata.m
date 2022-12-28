function [logF,logC] = logdata(global_best,fevals,input,global_old)
    global fcflag 
    global logF 
    global logC
    global logEvals
	budget = input.budget;
	if fevals > fcflag/10*budget && fcflag < 10;
		if ~isempty(global_old)
			if lex_rank(global_old.val,global_old.conv,global_best.val,global_best.conv)==0
				logf = global_best.val;	
				logc = global_best.conv;
			else
				logf = global_old.val;	
				logc = global_old.conv;
			end			
		else
			logf = global_best.val;	
			logc = global_best.conv;
        end
        logF(fcflag) = logf;
        logC(fcflag) = logc;
        logEvals(fcflag) =fevals;
		fcflag = fcflag +1;
	end
	
end

%% ============ Improved Multi-operator Differential Evolution Algorithm (IMODE) ============
% Should you have any queries, please contact
% Dr. Karam Sallam. Zagazig University
% karam_sallam@zu.edu.eg
% =========================================================================
 clc
format short e;
%%  introductory Definitions
num_prbs = 10;                      %% number of test problems
max_runs=30;                        %% number of runs
outcome=zeros(max_runs,1);          %% to save the solutions of each run
com_time=zeros(max_runs,1);         %% Computational time
SR=zeros(max_runs,1);               %% How many times the optimal solution is obtained
Avg_FES=zeros(max_runs,1);          %% average fitness evaluations
Final_results=zeros(num_prbs,8);    %% to save the final results

%% run on more than one processor
myCluster = parcluster('local');
myCluster.NumWorkers = 10;  % define how many processors to use

%% ========================= main loop ====================================
for I_fno= [1:10]
    Par= Introd_Par(I_fno); %% set of parameters
    sol=zeros(30*1,Par.n); %% the best solution vector of each run
    vv=[];
   parfor run=1:max_runs
        [outcome(run),com_time(run),SR(run), Avg_FES(run),res, sol(run,:)]=IMODE_main(run,I_fno);
        
        %% to print the convergence of ech run % set 0 if not
        if Par.Printing==1
            res= res- repmat(Par.f_optimal,1,size(res,2));
            res(res<=1e-08)=0; ss=size(res,2);
            endv=res(ss);
            if size(res,2)<Par.Max_FES
                res(size(res,2):Par.Max_FES)=endv;
            end
            vv(run,:)= res(1:Par.Max_FES);
        end
    end
    
    Final_results(I_fno,:)= [min(outcome),max(outcome),median(outcome), mean(outcome),std(outcome),mean(com_time),mean(SR),mean(Avg_FES)];
    
    disp(Final_results);
   
    %% save the results in a text
    save('results.txt', 'Final_results', '-ascii');
    
    %% fitness values at different levels of the optimization process
    %%% required by the competition
    if Par.Printing==1
        for k=1:16
            lim(k)=Par.n^(((k-1)/5)-3).*Par.Max_FES;
        end
        lim= ceil(lim);
%         lim= [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0].*Par.Max_FES;
        res_to_print= vv(:,lim);
        name1 = 'Results_Record\IMODE';
        name2 = num2str(I_fno);
        name3 = '_';
        name4 = num2str(Par.n);
        name5 = '.dat';
        f_name=strcat(name1,name2,name3,name4,name5);
        res_to_print=res_to_print';
        save(f_name, 'res_to_print', '-ascii');
        name1 = 'Results_Record\seeds_';
        f_name=strcat(name1,name2,name3,name4,name5);
        %% save the seeds used - if needed
%         myMatrix2= double(seed_run);
%         save(f_name, 'myMatrix2', '-ascii');
        
    end
end

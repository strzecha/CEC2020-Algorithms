
%%==========================================================================
% Should you have any queries, please contact
% Dr. Karam Sallam. Faculty of Computers and Informatics, Zagazig
% University, Egypt
% karam_sallam@zu.edu.eg
%==========================================================================
% function []= main_loop()
%%  introductory Definitions
clc
format SHORTENG;
num_prbs =57;                  %% number of test problems
max_runs=1;                    %% number of runs
% n=30;
% PopSize=25*n;
ar=[];
conv=[];
gra=[];
oper=[];
FES=[];
bestxx=[];
outcome=zeros(max_runs,1);
com_time=zeros(max_runs,1);
FS=zeros(max_runs,1);
Avg_FES=zeros(max_runs,1); %% average fitness evaluations to reach f*+0.0001
Final_results=zeros(num_prbs,13);

%-----------------------------------------------------------------------------------------------------
% matlabpool open 4
%% parallel code
% myCluster = parcluster('local');
% myCluster.NumWorkers = 12;  % define how many processors to use
%% main loop
for I_fno=1:num_prbs%[1:22,23:57]
    Par= Cal_par(I_fno);
    vv_f=[];
    vv_vio=[];
    for run=1:25
        
        %         [Par.o, Par.M]= load_o_m(Par.n,I_fno);
        [outcome(run),com_time(run),FS(run), Avg_FES(run), OutVio(run),res_f,res_vio]=EnMODE_Main(run,I_fno,Par);
        if Par.Printing==1
            res_f= res_f;
            res_vio=res_vio;
            %             res(res<=1e-08)=0;
            if size(res_f,2)<Par.max_eval
                res_f(size(res_f,2):Par.max_eval)=outcome(run);
                res_vio(size(res_vio,2):Par.max_eval)=OutVio(run);
                
                
            end
            vv_f(run,:)= res_f(1:Par.max_eval);
            vv_vio(run,:)= res_vio(1:Par.max_eval);
            
        end
    end
    [minn,idmin]=min(outcome);
    [maxx,idmax]=max(outcome);
    %     [medx,idmed]=median(outcome);
    
    Final_results(I_fno,:)= [minn,OutVio(idmin), median(outcome),OutVio(13),mean(outcome),mean(OutVio), maxx, OutVio(idmax), std(outcome),std(OutVio),mean(com_time),mean(FS),mean(Avg_FES)];
    %     end
    disp(Final_results);
    save('results.txt', 'Final_results', '-ascii');
    
    if Par.Printing==1
        lim= [ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0].*Par.max_eval;
        res_to_print= vv_f(:,lim);
        name1 = 'Results_Record\EnMODE_RC';
        name2 = num2str(I_fno);
        name3 = '_';
        name4 = num2str(Par.n);
        name5= '_F';
        name6 = '.txt';
        f_name=strcat(name1,name2,name3,name5,name6);
        res_to_print=res_to_print';
        save(f_name, 'res_to_print', '-ascii');
        
        name7= '_CV';
        
        res_to_print_vio=vv_vio(:,lim);
        res_to_print_vio=res_to_print_vio';
        f_name=strcat(name1,name2,name3,name7,name6);
        
        save(f_name, 'res_to_print_vio', '-ascii');
        
        
        
    end
    
end


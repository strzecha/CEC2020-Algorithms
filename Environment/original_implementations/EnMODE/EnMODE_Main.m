%%==========================================================================
% Should you have any queries, please contact
% Dr. Karam Sallam. Faculty of Computers and Informatics, Zagazig
% University, Egypt
% karam_sallam@zu.edu.eg
%==========================================================================

%% a loop to run all test problems, each one runs for 51 runs
function [outcome,com_time,FS,avgFE,OutVio,res_f,res_vio]= EnMODE_Main(run,I_fno,Par)
global initial_flag VAR
initial_flag = 0;

%% some defenition at each run
% ----------------------------------------------------------
thrshold=1e-4;
check=0;

avgFE=Par.max_eval;
iter=0;             %% current generation
% rand ('state', sum(100*clock))
RandStream.setGlobalStream (RandStream('mt19937ar','seed',run));

%% define variables
fitx = zeros(Par.PopSize,1);
vio_det= zeros(Par.PopSize,Par.gn+Par.hn);
g = zeros(Par.PopSize,max(1,Par.gn));
h = zeros(Par.PopSize,max(1,Par.hn));
Par.DELTA =ones(1,Par.hn)*0.0001;
Par.DELTAinq= zeros(1,Par.gn);
current_eval=0; %% current fitness evaluations
iter = iter+1;       %% Increase iter by 1
count_iter=0;        %% to be used with CS
PopSize=Par.PopSize;
gbfit=zeros(1,3);

%% Initalize x
x=repmat(Par.xmin,Par.PopSize,1)+repmat((Par.xmax-Par.xmin),Par.PopSize,1).*rand(Par.PopSize,Par.n);

[fitx,g, h]=cec20_func(x,I_fno,Par.n);
current_eval =current_eval+Par.PopSize;
for i=1:Par.PopSize
    %% evaluate constraints
    vio_det(i,:) = violation_2(g(i,:),h(i,:),Par.DELTAinq, Par.gn, Par.hn, Par.DELTA,2 );
end

[~,indd]= sort (mean(vio_det)); %% start with easy ones
cur_cons=max(1,ceil((Par.gn +Par.hn)/2));

%% update FES
viol1=sum(vio_det,2)/(Par.gn+Par.hn);
VioFit=[viol1,fitx];
[~,indmin]=sortrows(VioFit);
viol111=viol1';
fitxxx=fitx';
indmin=indmin';
res_f=min(repmat(fitxxx(indmin(1)),1,Par.PopSize),fitxxx);
res_vio=min(repmat(viol111(indmin(1)),1,Par.PopSize,1),viol1');


VAR0=median(viol1);
cp=(-log(VAR0)-6)/log(1-0.5);

max_delta=median(viol1);
Par.delta =max(ones(1,Par.hn).*1e-4, min(ones(1,Par.hn).*max_delta, mean(abs(h(1:ceil(Par.PopSize*0.5),:)))));
Par.DELTA=Par.delta;
if Par.gn>0
    Par.DELTAinq= zeros(1,Par.gn);
    Par.delta_ieq = max(zeros(1,Par.gn),mean(max(zeros(ceil(Par.PopSize*0.5),Par.gn),g(1:ceil(Par.PopSize*0.5),:))));
    
    tt= find (Par.delta_ieq>=1000);
    if isnan(tt)==0
        Par.DELTAinq(tt)=Par.delta_ieq(tt);
    end
    Par.delta_ieq=Par.DELTAinq;
end
%% update current fitness evaluation
op_original=[1 2];
op=op_original;


%% set the parameters for archive
memory_size=Par.PopSize;
Afactor=2;
archive_f= 0.5.*ones(memory_size,1);
archive_Cr= 0.2.*ones(memory_size,1);
hist_pos=1;
archive.NP = Afactor * Par.PopSize; % the maximum size of the archive
archive.pop = zeros(0, Par.n); % the solutions stored in te archive
archive.funvalues = zeros(0, 1); % the function value of the archived solutions
%% main loop
if current_eval<Par.max_eval
    stop_con=0;
end
probDE1=1./2 .* ones(1,2);
cons_cs=100;
start_time=cputime;
com_time=0;
rule_iter=0;
X=0;
while ~stop_con
    count_iter=count_iter+1; % to control CS
    % adjusting the threshold
    if X < 0.5
        VAR=VAR0*(1-X)^cp;
    else
        VAR=0;
    end
    %% update delta
    
    [Par] = Update_delta(Par,I_fno,current_eval);
    %% a counter
    if cur_cons<=Par.gn +Par.hn
        rule_iter=rule_iter+1;
    end
    
    if (mod(rule_iter,cons_cs)==0)&& cur_cons<(Par.gn+Par.hn)
        cur_cons= min(Par.gn +Par.hn,cur_cons+ceil((Par.gn +Par.hn)/2));
        rule_iter=1;
    end
    loc=indd(1:cur_cons);
    
    vio_det=[];
    %% ======= update vio. of each DE without calling cons. functions =====
    for j=1:PopSize
        vio_det(j,:) = violation_2(g(j,:),h(j,:),Par.DELTAinq, Par.gn, Par.hn, Par.DELTA,2 );
    end
    if cur_cons ==1
        viol1=vio_det(:,indd(loc));
    else
        viol1=sum(vio_det(:,indd(loc)),2)/(Par.gn+Par.hn);
    end
    
    
    %% DE1
    [x, fitx,viol1,archive,hist_pos,memory_size, archive_f,archive_Cr,current_eval,g,h,probDE1,res_f,res_vio] = ...
        EnMODE( x, fitx,archive,hist_pos,memory_size, archive_f,archive_Cr,Par,  current_eval, I_fno,viol1,iter,PopSize,op,count_iter,g,h,loc,indd,probDE1,res_f,res_vio);
    
    
    [x, fitx, viol1,g,h,PopSize]=...
        reduction(x, fitx, viol1,g,h,Par.InitPop,Par.max_eval,Par.MinPopSize,PopSize,current_eval);
    
    %     [x,fitx,viol1,g,h,current_eval]=diversity(x,fitx,viol1,g,h,Par.xmin,Par.xmax,I_fno,current_eval,Par);
    
    
    if (size(find ((viol1==0)),1)>0)
        vv= find (viol1==0);
        [~,position]  = min(fitx(vv));
        position=vv(position);
    else
        [~,position]  = min(viol1);
    end
    
    %% ============= to plot convergence if needed
    X=X+current_eval/Par.max_eval;
    
    gbfit=[gbfit; current_eval fitx(position)  viol1(position)];
    
    %% a faster call
    %% a faster call
    
    test_vio(1,:) = violation_2(g(position,:),h(position,:),Par.DELTAinq, Par.gn, Par.hn, Par.DELTA,1 );
    vio_chk_opt=sum(test_vio,2);
    
    %% check to stop
    if (current_eval>=Par.max_eval)
        stop_con=1;
    else
        stop_con=0;
    end
    
    VioFit=[viol1,fitx];
    [~,indmin]=sortrows(VioFit);
    viol111=viol1';
    fitxxx=fitx';
    indmin=indmin';
    % res_f=[res_f repmat(fitxxx(indmin(1)),1,PopSize)];
    % res_vio=[res_vio repmat(viol111(indmin(1)),1,PopSize,1)];
    %% check if the optimal is reached
    
    if ( (abs ( fitx(position))<= thrshold) && vio_chk_opt(1)==0 && check==0)
        
        avgFE=current_eval;
        if check==0
            com_time= cputime-start_time;
        end
        check=1;
    end
    
    %% print out data
    if stop_con
        if check==0
            com_time= cputime-start_time;
        end
        
        fprintf('run\t %d, fitness\t %d, violation\t %d, avg.FFE\t %d\n', run, fitx(position), viol1(position),avgFE);
        outcome= fitx(position);
        FS= (viol1(position)==0);
        OutVio=viol1(position);
        x_best=x(position,:);
        
        iter = iter+1;
        
    end
    
end


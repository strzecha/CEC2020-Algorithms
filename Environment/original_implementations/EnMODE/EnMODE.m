
%%==========================================================================
% Should you have any queries, please contact
% Dr. Karam Sallam. Faculty of Computers and Informatics, Zagazig
% University, Egypt
% karam_sallam@zu.edu.eg
%==========================================================================

function[x, fitx,vio,archive,hist_pos,memory_size, archive_f,archive_Cr,current_eval,g,h,prob,res_f,res_vio] = EnMODE( x, fitx,archive,hist_pos,memory_size, archive_f,archive_Cr,Par,current_eval,...
    I_fno,vio,iter,PopSize,opp,count_iter,g,h,loc,indd,prob,res_f,res_vio)

global VAR
fitx_new = zeros(PopSize,1);
mem_rand_index = ceil(memory_size * rand(PopSize, 1));
mu_sf = archive_f(mem_rand_index);
mu_cr = archive_Cr(mem_rand_index);

% ========================= generate CR ==================================
cr = mu_cr + 0.1 * tan(pi * (rand(PopSize,1) - 0.5));
pos = find(cr <= 0);
while ~ isempty(pos)
    cr(pos) = mu_cr(pos) + 0.1 * tan(pi * (rand(length(pos),1) - 0.5));
    pos = find(cr <= 0);
end
cr = min(cr, 1);
%
% %% ========================= generate F ===================================
F = mu_sf + 0.1 * tan(pi * (rand(PopSize,1) - 0.5));
pos = find(F <= 0);
while ~ isempty(pos)
    F(pos) = mu_sf(pos) + 0.1 * tan(pi * (rand(length(pos),1) - 0.5));
    pos = find(F <= 0);
end
F = min(F, 1);



% pop=x;
popAll = [x;archive.pop]; %% set archive
r0 = 1 : PopSize;
[r1, r2,r3,r4] =gn_rand(PopSize, size(popAll,1));
[~, sorted_index] = sort(fitx, 'ascend');

bb= rand(PopSize, 1);
probiter = prob(1,:);
l2= sum(prob(1:2));
op_1 = bb <=  probiter(1)*ones(PopSize, 1);
op_2 = bb > probiter(1)*ones(PopSize, 1) &  bb <= (ones(PopSize, 1)) ;
% op_3 = bb > l2*ones(PopSize, 1) &  bb <= (ones(PopSize, 1)) ;
len=length(opp);
switch len
    case 3
        vccc=ceil(PopSize/len);
        operator(1:vccc)=opp(1);
        operator(vccc+1:2*vccc)=opp(2);
        operator(PopSize-vccc:PopSize)=opp(3);
        rnd=randperm(PopSize,PopSize);
        operator=operator(rnd);
        
    case 2
        vccc=ceil(PopSize/len);
        operator(1:vccc)=opp(1);
        operator(PopSize-vccc:PopSize)=opp(2);
        rnd=randperm(PopSize,PopSize);
        operator=operator(rnd);
        
    case 1
        operator(1:PopSize)=opp(1);
        rnd=randperm(PopSize,PopSize);
        operator=operator(rnd);
        
end
for k=1:3
    v_op(k,:)=(operator==k);
end

vi=zeros(PopSize,Par.n);

%% current-to-pbest/archive
% if max(v_op(1,:))==1
p_best_rate=0.25;
pNP = max(round(p_best_rate * PopSize), 2); %% choose at least two best solutions
randindex = ceil(rand(1, PopSize) .* pNP); %% select from [1, 2, 3, ..., pNP]
randindex = max(1, randindex); %% to avoid the problem that rand = 0 and thus ceil(rand) = 0
phix = x(sorted_index(randindex), :); %% randomly choose one of the top 100p% solutions

% end
vi(op_1==1,:) = x(op_1==1,:)+ F(op_1==1, ones(1, Par.n)) .*(phix(op_1==1,:) - x(op_1==1,:) + x(r1(op_1==1), :) - popAll(r3(op_1==1), :));
vi(op_2==1,:) =  x(op_2==1,:)+ F(op_2==1, ones(1, Par.n)) .*(phix(op_2==1,:) - x(op_2==1,:) + x(r1(op_2==1), :) - x(r2(op_2==1), :));
%% handeling Boundry
% vi = han_boun(vi,pop,Par,PopSize);
vi = han_boun(vi, Par.xmax, Par.xmin, x,PopSize,2);
%% crossover and selection
mask = rand(PopSize, Par.n) > cr(:, ones(1, Par.n)); % mask is used to indicate which elements of ui comes from the parent
rows = (1 : PopSize)';
cols = floor(rand(PopSize, 1) * Par.n)+1; % choose one position where the element of ui doesn't come from the parent
jrand = sub2ind([PopSize Par.n], rows, cols);
mask(jrand) = false;
ui = vi;
ui(mask) = x(mask);
%% ==================== Preallocating Arrays to reduce time =================
fitx_new = zeros(PopSize,1);
vio_new = zeros(PopSize,1);
vio_det_new= zeros(PopSize,Par.gn+Par.hn);
g_new = zeros(PopSize,max(1,Par.gn));
h_new = zeros(PopSize,max(1,Par.hn));
[fitx_new,g_new, h_new]=cec20_func(ui,I_fno,Par.n);
for i=1:PopSize
    vio_det_new(i,:) = violation_2(g_new(i,:),h_new(i,:),Par.DELTAinq, Par.gn, Par.hn, Par.DELTA,2 );
    current_eval =current_eval+1;
    
end
vio_new=sum(vio_det_new(:,indd(loc)),2)/(Par.gn+Par.hn);
I=pairwise_comp (fitx_new,fitx,vio_new,vio,0);

%% ===========================  calc. Improv ==============================
diff_vio=zeros(1,PopSize);
I2I = find(vio_new>0 & vio>0 & I==1);
diff_vio(I2I) = max(0,(vio(I2I) - vio_new(I2I))./vio(I2I))...
    + max(0,(fitx(I2I) - fitx_new(I2I))./abs(fitx(I2I))) ; %% Inf. to Inf.

IF2F= find((vio_new==0 & vio==0) | (vio_new==0 & vio>0)); %% Inf. or Fes. to Fes.
diff_f=zeros(1,PopSize);
val=((abs(fitx(IF2F))));
val(val==0)=1;
diff_f(IF2F) = max(diff_vio) + max(0,(vio(IF2F) - vio_new(IF2F))./vio(IF2F))...
    + max(0,(fitx(IF2F) - fitx_new(IF2F))./val);%% we only consider successful ones later

diff= diff_f+diff_vio;
Sucrate=sum(I==0)/PopSize;


vio(I == 1, :) = vio_new(I == 1, :);
fitx(I == 1, :) = fitx_new(I == 1, :);
h(I == 1, :) = h_new(I == 1, :);
g(I == 1, :) = g_new(I == 1, :);
goodCR = cr(I == 1);
goodF = F(I == 1);
%% ========================= update archive ===============================
archive = updateArchive(archive, x(I == 0, :), fitx(I == 0)');
x(I == 1, :) = ui(I == 1, :);

diff2 = max(0,diff);
z1=(op_1==1)';
z2=(op_2==1)';
diff11=diff2(z1);
diff22=diff2(z2);
EA1=x(v_op(1,:),:);
EA2=x(v_op(2,:),:);
if current_eval<= Par.limit_all
    %% compute the success diversity of each operator
%     D=zeros(2,1);
%     D(1) = mean(pdist2(EA1(2:size(EA1,1),:),EA1(1,:)));
%     D(2) = mean(pdist2(EA2(2:size(EA2,1),:),EA2(1,:)));
%     %     D(3) = mean(pdist2(EA3(2:size(EA3,1),:),EA3(1,:)));
%     norm_div= D./sum(D);
%     D=norm_div;
    %% ==================== update Prob. of each DE ===========================
    
    count_S(1)=max(0,mean(diff11));
    count_S(2)=max(0,mean(diff22));
    %  count_S(3)=max(0,mean(diff33));
    
else
    count_S=0;
end

if count_S~=0
    prob= max(0.1,min(0.9,count_S./(sum(count_S))));
else
    prob=1/2* ones(1,2);
end
%% =================== update memory cr and F =============================
num_success_params = numel(goodCR);
if num_success_params > 0
    weightsDE = diff(I ==1)./ sum(diff(I ==1));
    %% for updating the memory of scaling factor
    archive_f(hist_pos) = (weightsDE * (goodF .^ 2))./ (weightsDE * goodF);
    
    %% for updating the memory of crossover rate
    if max(goodCR) == 0 || archive_Cr(hist_pos)  == -1
        archive_Cr(hist_pos)  = -1;
    else
        archive_Cr(hist_pos) = (weightsDE * (goodCR .^ 2)) / (weightsDE * goodCR);
    end
    
    hist_pos= hist_pos+1;
    if hist_pos > memory_size;  hist_pos = 1; end
    
end

%% check to print
VioFit=[vio,fitx];
[~,indmin]=sortrows(VioFit);
viol111=vio';
fitxxx=fitx';
indmin=indmin';
res_f=[res_f repmat(fitxxx(indmin(1)),1,PopSize)];
res_vio=[res_vio repmat(viol111(indmin(1)),1,PopSize,1)];



[~,loc]=min(fitx);

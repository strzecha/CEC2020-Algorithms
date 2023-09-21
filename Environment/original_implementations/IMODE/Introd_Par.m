%% ============ Improved Multi-operator Differential Evolution Algorithm (IMODE) ============
% Should you have any queries, please contact
% Dr. Karam Sallam. Zagazig University
% karam_sallam@zu.edu.eg
% =========================================================================
% Some part of this code is taken from UMOEA-II
% =========================================================================
function [Par] = Introd_Par(I_fno)

%% loading

Par.n_opr=3;  %% number of operators 
Par.n=15;     %% number of decision vriables


if Par.n==5
    Par.CS=100; %% cycle
    Par.Max_FES=50000;
    Par.Gmax = 2163;
elseif Par.n==10
    Par.CS=100; %% cycle
    Par.Gmax = 2745;
    Par.Max_FES=1000000;
elseif Par.n==15
    Par.CS=100; %% cycle
    Par.Gmax = 3022;
    Par.Max_FES=3000000;
else
    Par.Max_FES=10000000;
    Par.CS=100; %% cycle
    Par.Gmax = 3401;
end
opt= [100, 1100,700,1900,1700,1600,2100,2200,2400,2500];      %% define the optimal solution as shown in the TR
Par.xmin= -100*ones(1,Par.n);
Par.xmax= 100*ones(1,Par.n);

Par.f_optimal=opt(I_fno);
Par.PopSize=6*Par.n*Par.n; %% population size
% Par.PopSize=18*Par.n; %% population size

Par.MinPopSize=4;
Par.MinPopSize1=4+floor(3*log(Par.n));

Par.prob_ls=0.1;
%% printing the detailed results- this will increase the computational time
Par.Printing=1; %% 1 to print; 0 otherwise

end
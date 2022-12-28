
%============Adaptive Configuration of DEs for big data==========
%%==========================================================================
% Should you have any queries, please contact
% Dr. Saber Elsayed. University of New South Wales at Canberra
% s.elsayed@adfa.edu.au
% https://sites.google.com/site/saberelsayed3/home
%==========================================================================
function [] = PrintSummaryFile(I_fno,sum_res,SingMul)
list = ['D4  '; 'D12 '; 'D19 '; 'D4N '; 'D12N'; 'D19N'];
namePart1 = 'Karam_';
namePart2 = num2str(list(I_fno,:));
namePart4 = '.dat';
        namePart0 = 'Sing_res_Summary/';
        namePart3 = '_Sing_Sum';
        f_name=strcat(namePart0,namePart1,namePart2,namePart3,namePart4);
        save(f_name, 'sum_res', '-ascii');
   
end
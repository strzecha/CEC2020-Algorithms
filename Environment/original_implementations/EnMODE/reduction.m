function [x, fitx, vio,g,h,PopSize]= ...
    reduction(x, fitx, vio,g,h,Ori_PopSize,Max_FES,min_pop_size,PopSize,current_eval)
 plan_pop_size = round((((min_pop_size - Ori_PopSize) / Max_FES) ...
        * current_eval) + Ori_PopSize);
    if PopSize > plan_pop_size
        reduction_ind_num = PopSize - plan_pop_size;
        if PopSize - reduction_ind_num <  min_pop_size
            reduction_ind_num = PopSize - min_pop_size;
        end
        for r = 1 : reduction_ind_num
            violFitness=[vio,fitx];
        [~,indViol]=sortrows(violFitness);
        vv=indViol(PopSize);
%              vv =ceil(0.95*PopSize)+randi(PopSize-ceil(0.95*PopSize));
%               x_arch=[x_arch;x(vv,:)];
            fitx(vv)=[];
            vio(vv)=[];
            x(vv,:)=[];
            g(vv,:)=[];
            h(vv,:)=[];
            PopSize = PopSize - 1;
        end
    end
end

% UpdPopSize = round((((Par.MinPopSize - Par.InitPop) / Par.max_eval) * current_eval) + Par.InitPop);
% if PopSize > UpdPopSize
%     reduction_ind_num = PopSize - UpdPopSize;
%     if PopSize - reduction_ind_num <  Par.MinPopSize
%         reduction_ind_num = PopSize - Par.MinPopSize;
%     end
%     %% remove the worst ind.
%     for r = 1 : reduction_ind_num
%         violFitness=[viol1,fitx];
%         [~,indViol]=sortrows(violFitness);
%         vv=indViol(PopSize);
%         x(vv,:)=[];
%         fitx(vv)=[];
%         viol1(vv)=[];
%         PopSize = PopSize - 1;
%     end
%     archive.NP = round(1.4 * PopSize);
%     if size(archive.pop, 1) > archive.NP
%         rndpos = randperm(size(archive.pop, 1));
%         rndpos = rndpos(1 : archive.NP);
%         archive.pop = archive.pop(rndpos, :);
%     end
% end
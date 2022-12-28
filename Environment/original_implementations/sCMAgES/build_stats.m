function [out]=build_stats(FitT,input)

r      = eps_sort(FitT(:,11),FitT(:,12),0);
Fit    = FitT(r, 11);
Con    = FitT(r, 12);
bestF  = Fit(1);
bestC  = Con(1);
medF   = Fit((input+1)/2);
medC   = Con((input+1)/2);
meanF  = mean(Fit);
meanC  = mean(Con);
worstF = Fit(input);
worstC = Con(input);
stdF   = std(Fit);
stdC   = std(Con);
FR     = sum(Con == 0)./input*100;
c_1    = FitT(r((input+1)/2),13);
c_2    = FitT(r((input+1)/2),14);
c_3    = FitT(r((input+1)/2),15);

%% my code for mean calculation
out = [ bestF bestC medF medC meanF meanC worstF worstC stdF stdC FR c_1 c_2 c_3];
end
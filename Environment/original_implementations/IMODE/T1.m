n = 10;
run=0;                          %% Set run at 0
 pop_size=200000;                    %% population size
I_fno=18;
xmax = 100.0;
xmin = -100.0;
popold = repmat(xmin.*ones(1,n), pop_size, 1) + rand(pop_size, n) .* (repmat(xmax.*ones(1,n) - xmin.*ones(1,n), pop_size, 1));
tic
fitness = cec17_func(popold',I_fno);
toc
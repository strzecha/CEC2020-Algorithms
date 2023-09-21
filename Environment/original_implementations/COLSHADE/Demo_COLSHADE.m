clear;
clc;

test_function = 1;              % CEC2020 RC problem [1, 57]
% record = true;                % For recording the best-found solution every 10% of FEs
record = false;                 % For only display the best-found solution every 10% of FEs
prob_levy = 0.5;                % Initial probability of Levy flight based mutation
dynamic_tolerance = true;       % Using dynamic tolerance for equality constraints.
% dynamic_tolerance = false;    % Fix tolerance for equality constraints to 10^{-4}.

[x, f, g, h] = colshade(test_function, record, prob_levy, dynamic_tolerance)
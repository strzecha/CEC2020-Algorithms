function [x_opt, f_opt, g_opt, h_opt] = colshade(func_num, save_progress, prob_levy, dynamic)
	% File for recording results similar to the requested format:
	file_name = sprintf("%02d.txt", func_num);

	global initial_flag
	initial_flag = 0;
	rand('state', sum(100 * clock))

	pd_normal = makedist('Normal', 'mu', 0, 'sigma', 0.1);
	pd_cauchy = makedist('Stable', 'alpha', 1.0, 'beta', 0, 'gam', 0.1, 'delta', 0);
	pd_levy = makedist('Stable','alpha', 0.5, 'beta', 1, 'gam', 0.01, 'delta', 0.01);

	%% Problem parameters
	par = Cal_par(func_num);	% Optimization problem parameters:
	D = par.n;					% Dimensionality
	gn = par.g;					% Number of inequality constraints
	hn = par.h;					% Number of equality constraints
	xmin = par.xmin;			% Lower bound of the variables
	xmax = par.xmax;			% Upper bound of the variables

	% fprintf("D: %d ng: %d nh: %d\n", D, gn, hn);	% Show problem deatails

	%% Population parameters
	r_init = 18;				% Rate of initial population size with respect to D.
	r_arc = 2.6;				% Rate of population size with respect to population size.

	max_ps = ceil(r_init * D);
	ps = max_ps;				% Population size at iteration i.
	min_ps = 4;					% Min population size.
	pbest_rate = 0.11;			% Percent of solutions consider as p_best.

	%% Handling of constraints and mutations
	momentum = 0.25;			% Momentum for update prob_levy
	prob_min = 1e-3;			% Min probability for both mutations
	tolerance = 1e-4;			% Final tolerance for equality constraints (Fixed according the competence)
	cut_off = 0.6;				% Proportion of FEs before drop dynamic tolerance value

	%% Memory parameters:
	% Current-to-pbest
	H = 6;						% Size of historical memory
	M_F = 0.5 * ones(1, H);		% Cyclic memory for mutation factor
	M_CR = 0.5 * ones(1, H);	% Cyclic memory for crossover rate
	k = 1;						% Index for cyclic memory update

	% Levy flight
	M_F_l = 0.5 * ones(1, H);	% Cyclic memory for scale factor
	M_CR_l= 0.5 * ones(1, H);	% Cyclic memory for crossover rate
	k_l = 1;                    % Index for cyclic memory update

	%% Maximum Function Evaluations and Checkpoints
	if D <= 10
		max_fes = 1e5;
	elseif D <= 30
		max_fes = 2e5;
	elseif D <= 50
		max_fes = 4e5;
	elseif D <= 150
		max_fes = 8e5;
	else
		max_fes = 1e6;
	end

	%% Checkpoints for save/display current results.
	checkpoints = linspace(0.1, 1, 10);
	checkpoints = max_fes * checkpoints;
	check_index = 1;

	%% Initialize population and archive
	x = xmin + rand(ps, D) .* (xmax - xmin);
	x_arch= [];
	as = 0;

	[f, g, h] = cec20_func(x, func_num);
	g = g';								% cec20_func transposes constraints
	h = h';								% Re-transpose for keep consistence
	fes = ps;
	[svc, ~] = get_svc(g, h, 0.);
	[x, f, g, h, ~, ~] = sort_pop(x, f, g, h, svc, svc);

	if dynamic
		tolerance_i = max(abs(h));
		tolerance_i = max(tolerance_i, tolerance);
	else
		tolerance_i = tolerance;
	end
	cut_off = cut_off * max_fes;
	[svc, ~] = get_svc(g, h, tolerance_i);
	[svc_abs, ~] = get_svc(g, h, tolerance);
	[x, f, g, h, ~, svc_abs] = sort_pop(x, f, g, h, svc, svc_abs);

	%% Evolutionary process
	while fes + ps <= max_fes
		%% Update tolerance
		[svc, h_sum] = get_svc(g, h, tolerance_i);
		if any(h_sum==0) && any(tolerance_i > tolerance) && (fes < cut_off)
			tolerance_i = get_tolerance(h, tolerance_i, tolerance, cut_off - fes, h_sum);
			[svc, ~] = get_svc(g, h, tolerance_i);
		end

		if (fes >= cut_off) && any(tolerance_i > tolerance)
			tolerance_i = tolerance;				% Changing from array to scalar
			[svc, ~] = get_svc(g, h, tolerance_i);
		end

		%% Mutation and crossover
		[r, p_best] = get_index(ps, as, pbest_rate);
		[F, CR, pbest_flag] = get_cx_params(M_F, M_CR, M_F_l, M_CR_l, prob_levy, ps, pd_normal, pd_cauchy);
		u = crossover(x, x_arch, r, p_best, F, CR, pbest_flag, xmin, xmax, pd_levy);
		[f_u, g_u, h_u]	= cec20_func(u, func_num);
		g_u = g_u';
		h_u = h_u';
		[svc_u, ~] = get_svc(g_u, h_u, tolerance_i);
		[svc_u_abs, ~] = get_svc(g_u, h_u, tolerance);
		fes = fes + size(u, 1);

		%% Update population and memory
		[x, f, g, h, svc, svc_abs, delta_f, x_arch] = deb_tournament(...
			x, f, g, h, svc, svc_abs, u, f_u, g_u, h_u, svc_u, svc_u_abs,...
			x_arch, tolerance_i, tolerance);

		[M_F, M_CR, M_F_l, M_CR_l] = update_memory(M_F, M_CR, M_F_l, ...
			M_CR_l, F, CR, pbest_flag, delta_f, k, k_l);

		[x, f, g, h, svc, svc_abs] = sort_pop(x, f, g, h, svc, svc_abs);

		delta_pbest = delta_f .* pbest_flag;
		delta_levy  = delta_f .* (~pbest_flag);

		if any(delta_pbest)
			k = k + 1;
			if k > H
				k = 1;
			end
		end

		if any(delta_levy)
			k_l = k_l + 1;
			if k_l > H
				k_l = 1;
			end
		end

		if any(delta_pbest > 0) || any(delta_levy > 0)
			prob_levy = momentum * prob_levy + (1 - momentum) * ...
				(sum(delta_levy) / (sum(delta_levy) + sum(delta_pbest)));
			prob_levy = max(min(prob_levy, 1 - prob_min), prob_min);
		end

		%% Resize population and archive size
		ps = round(min_ps + (1 - (fes / max_fes)) * (max_ps - min_ps)) ;
		x = x(1:ps, :);
		f = f(1:ps);
		g = g(1:ps, :);
		h = h(1:ps, :);
		svc = svc(1:ps);
		svc_abs	= svc_abs(1:ps);

		max_as = round(ps * r_arc);
		as = size(x_arch, 1);

		if as > max_as
			idx = randperm(as);
			idx = idx(1:as - max_as);
			x_arch(idx, :) = [];
			as = size(x_arch, 1);
		end

		%% Record current results
		if fes >= checkpoints(check_index)
			if save_progress
				to_record = [x(1,:), f(1), g(1, :), h(1, :)];
				save("-append", "-ascii", file_name, "to_record");
			else
				% Print only if not saving to file
				fprintf("FES: %d / %d f_best: %d svc_best: %d prob_levy: %d tolerance: %d\n", ...
					fes, max_fes, f(1), svc_abs(1), prob_levy, max(tolerance_i));
			end
			check_index = check_index + 1;
		end
	end

	% Best found solution
	x_opt = x(1,:);
	f_opt = f(1);
	g_opt = g(1,:);
	h_opt = h(1,:);
	svc_opt	= svc_abs(1);

	%% Record results
    if check_index <= size(checkpoints, 2)
        if save_progress
            to_record = [x_opt, f_opt, g_opt, h_opt];
            save("-append", "-ascii", file_name, "to_record");
        else
            % Print only if not saving to file
            fprintf("FES: %d / %d f_best: %d svc_best: %d prob_levy: %d tolerance: %d\n", ...
                fes, max_fes, f_opt, svc_opt, prob_levy, max(tolerance_i));
        end
    end
end

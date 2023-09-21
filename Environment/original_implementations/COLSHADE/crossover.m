function [u] = crossover(x, x_arch, r, pbest, F, CR, pbest_flag, xmin, xmax, pd_levy)
	% Mutation
	[ps, D] = size(x);
	v = zeros(ps, D);

	for i = 1:ps
		p  = pbest(i);
		if pbest_flag(i)				% current-to-pbest
			r1 = r(i, 1);
			r2 = r(i, 2);
			if r2 > ps 					% using solution in file
				r2 = r2 - ps;
				v(i, :) = x(i, :) + F(i) * (x(p, :) - x(i, :)) + F(i) * (x(r1, :) - x_arch(r2, :));
			else
				v(i, :) = x(i, :) + F(i) * (x(p, :) - x(i, :)) + F(i) * (x(r1, :) - x(r2, :));
			end
		else 							% levy flight
			levy_rand = pd_levy.random(1, D);
			v(i, :) = x(i, :) + F(i) * levy_rand .* (x(p, :) - x(i, :));
		end
	end

	% Bound constraint
	v_upper = v > xmax;
	v_lower = v < xmin;

	v = v.*(~v_upper);				% Make zero outbounded values
	v = v.*(~v_lower);

	base = 0.1 * rand(ps, D);
	x_upper = (1 - base) .* xmax + base .* x;
	x_lower = (1 - base) .* xmin + base .* x;

	v_upper = v_upper .* x_upper;	% Keep x_{i,j} values where v_{i,j} is outbounded
	v_lower = v_lower .* x_lower;

	v = v + v_upper + v_lower;

	% Crossover (binomial)
	cross_  = rand(ps, D);
	for i = 1:ps
		cross_(i, randi(D)) = 0;	% get j_rand value for crossover
	end

	v_values = cross_ <= CR;
	x_values = cross_ > CR;
	v = v .* v_values;
	x = x .* x_values;
	u = v + x;
end
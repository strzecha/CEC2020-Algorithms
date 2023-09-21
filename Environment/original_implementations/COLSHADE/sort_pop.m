function [x, f, g, h, svc, svc_abs] = sort_pop(x, f, g, h, svc, svc_abs)
	% By sum of violated constraints
	[svc_abs, idx]= sort(svc_abs);
	x = x(idx, :);
	f = f(idx);
	g = g(idx, :);
	h = h(idx, :);
	svc = svc(idx);

	% By fitness function in feasible solutions
	ps = length(f);
	end_ = 1;
	for i = 1:ps
		if svc_abs(i) > 0.
			break
		end
		end_ = i;
	end

	[~, idx] = sort(f(1:end_));
	x(1:end_, :) = x(idx, :);
	f(1:end_) = f(idx);
	g(1:end_, :) = g(idx, :);
	h(1:end_, :) = h(idx, :);
	svc(1:end_) = svc(idx);
	svc_abs(1:end_)	= svc_abs(idx);
end
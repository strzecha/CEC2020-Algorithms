function [x, f, g, h, svc, svc_abs, delta_f, x_arch] = deb_tournament(x, f, g, h, ...
	svc, svc_abs, u, f_u, g_u, h_u, svc_u, svc_u_abs, x_arch, tolerance_i, tolerance_f)

	ps = size(x, 1);
	delta_f = zeros(ps, 1);
	delta_svc = zeros(ps, 1);

	for i = 1:ps 
		if i == 1							% preserve or improve current best individual
			if svc_u_abs(i) < svc_abs(i)
				delta_svc(i) = svc_abs(i) - svc_u_abs(i);
			elseif (svc_u_abs(i) == 0) && (svc_abs(i) == 0) && (f_u(i) < f(i))
				delta_f(i) = f(i) - f_u(i);
			end
		else
			if svc_u(i) < svc(i)					% Accept currently feasible over currently infeasible (improves diversity)
				delta_svc(i) = svc(i) - svc_u(i);
			elseif (svc_u(i) == 0) && (svc(i) == 0) && (f_u(i) < f(i))	% Accept currently optimal over currently sub-optimal
				delta_f(i) = f(i) - f_u(i);
			elseif any(tolerance_i > tolerance_f) && (svc_u_abs(i) <  svc_abs(i))	% Accept more feasible over infeasible
				delta_svc(i) = svc_abs(i) - svc_u_abs(i);
			end
		end
	end

	% Normalization to make comparable
	delta_f_max = max(delta_f);
	delta_svc_max = max(delta_svc);

	if delta_f_max > 0
	 	delta_f = delta_f / delta_f_max;
	end
	if delta_svc_max > 0
	 	delta_svc = delta_svc / delta_svc_max;
	end

	delta_f = delta_f + delta_svc;

	for i = 1:ps
		if delta_f(i) > 0
			x_arch = [x_arch; x(i, :)];
			x(i,:) = u(i, :);
			g(i,:) = g_u(i, :);
			h(i,:) = h_u(i, :);
			f(i) = f_u(i);
			svc(i) = svc_u(i);
			svc_abs(i) = svc_u_abs(i);
		end
	end
end

function [svc, h_sum] = get_svc(g, h, epsilon)
	g = max(g, 0);
	h = max(abs(h) - epsilon, 0);
	svc = sum(g, 2) + sum(h, 2);
    h_sum = min(h);					% for evaluating feasibility independently
end
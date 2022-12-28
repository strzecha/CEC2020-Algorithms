function [r, p_best] = get_index(ps, as, p_best_percent)
	p_best = randi(int32(ceil(p_best_percent * ps)), ps, 1);
	r = zeros(ps, 2);

	for i = 1:ps
		while true
			r(i, 1) = randi(ps);
			if r(i, 1) ~= i
				break;
			end
		end

		while true
			r(i, 2) = randi(ps + as);
			if r(i, 2) ~= i && r(i, 2) ~= r(i, 1)
				break;
			end
		end
	end
end
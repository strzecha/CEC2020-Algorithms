function [tolerance_i] = get_tolerance(h, tolerance_i, tolerance_f, fes, h_sum)
	ps = size(h, 1);
    h_sum = h_sum == 0;
	decay = (tolerance_f ./ tolerance_i) .^ (ps / fes);
    decay = decay .* h_sum;
    decay = decay + (~h_sum);
	tolerance_i = tolerance_i .* decay;
	tolerance_i = max(tolerance_i, tolerance_f); 
end
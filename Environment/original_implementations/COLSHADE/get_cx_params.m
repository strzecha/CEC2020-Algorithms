function [F, CR, pbest_flag] = get_cx_params(M_F, M_CR, M_F_l, M_CR_l, prob_levy, ps, pd_normal, pd_cauchy)
	H = length(M_F);
	F = zeros(ps, 1);
	CR = zeros(ps, 1);
	pbest_flag = zeros(ps, 1);
	p_i_levy = rand(ps, 1);

	for i = 1:ps
		r = randi(H);
		if p_i_levy(i) <= prob_levy
			m_cr = M_CR_l(r);
			m_f = M_F_l(r);  
		else
			m_cr = M_CR(r);
			m_f = M_F(r);
			pbest_flag(i) = 1;
		end
		if m_cr ~= -1
			CR(i) = m_cr + pd_normal.random;
		end
		
		F_crit  = sqrt((1 - CR(i) / 2) / ps);
		while F(i) <= F_crit
			F(i) = m_f + pd_cauchy.random;
		end
	end

	CR = min(max(CR, 0), 1);
	F = min(F, 1.);
end
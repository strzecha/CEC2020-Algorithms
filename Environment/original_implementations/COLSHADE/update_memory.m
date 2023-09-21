function [M_F, M_CR, M_F_l, M_CR_l] = update_memory(M_F, M_CR, M_F_l, ...
	M_CR_l, F, CR, pbest_flag, delta_f, k, k_l)

	ps = size(F, 1);

	S_F = [];
	S_CR = [];
	delta_pbest = [];

	S_F_l = [];
	S_CR_l = [];
	delta_levy = [];

	for i = 1:ps
		if delta_f(i) > 0
			if pbest_flag(i) == 1
				S_F = [S_F; F(i)];
				S_CR = [S_CR; CR(i)];
				delta_pbest = [delta_pbest; delta_f(i)];
			else
				S_F_l = [S_F_l; F(i)];
				S_CR_l = [S_CR_l; CR(i)];
				delta_levy = [delta_levy; delta_f(i)];
			end
		end
	end

	if ~isempty(S_F)
		w = delta_pbest / sum(delta_pbest);
		if (M_CR(k) == -1) || (max(S_CR) == 0)
			M_CR(k) = -1;
		else
			M_CR(k) = sum(w .* (S_CR .* S_CR)) / sum(w .* S_CR);
		end
		M_F(k) = sum(w .* (S_F .* S_F)) / sum(w .* S_F);
	end

	if ~isempty(S_F_l)
		w = delta_levy / sum(delta_levy);
		if (M_CR_l(k_l) == -1) || (max(S_CR_l) == 0)
			M_CR_l(k_l) = -1;
		else
			M_CR_l(k_l) = sum(w .* (S_CR_l .* S_CR_l)) / sum(w .* S_CR_l);
		end
		M_F_l(k_l) = sum(w .* (S_F_l .* S_F_l)) / sum(w .* S_F_l);
	end
end
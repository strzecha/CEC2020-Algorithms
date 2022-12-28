function [ui_y]=keep_range(ui_y,lower_bounds,upper_bounds)
	bwidth =  upper_bounds -  lower_bounds;
    n=length(ui_y);
	for j=1:n
        if ui_y(j)< lower_bounds(j)
		exceed =  lower_bounds(j)-ui_y(j);
			if exceed >= bwidth(j)
			    exceed = exceed - floor(exceed/bwidth(j))*bwidth(j);
			end
		    ui_y(j) =  lower_bounds(j) + exceed;
		    
		elseif ui_y(j) >  upper_bounds(j)
		    exceed = ui_y(j)-upper_bounds(j);
			if exceed >= bwidth(j)
			    exceed = exceed - floor(exceed/bwidth(j))*bwidth(j);
			end
		    ui_y(j) = upper_bounds(j) - exceed; 
		    
		end       
	end
end

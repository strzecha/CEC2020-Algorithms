function [ui_y] = keep_range(ui_y,lower_bounds,upper_bounds,kk)
bwidth =  upper_bounds -  lower_bounds;        
[n,m] = size(ui_y);
if kk == 1
for i = 1:m
	for j=1:n
        if ui_y(j,i)< lower_bounds(j)                                    
            exceed =  lower_bounds(j)-ui_y(j,i);  
			if exceed >= bwidth(j)
			    exceed = exceed - floor(exceed/bwidth(j))*bwidth(j);    
            end
		    ui_y(j,i) =  lower_bounds(j) + exceed;                       
		elseif ui_y(j,i) >  upper_bounds(j)                              
		    exceed = ui_y(j,i)-upper_bounds(j);
			if exceed >= bwidth(j)
			    exceed = exceed - floor(exceed/bwidth(j))*bwidth(j);  
            end
		    ui_y(j,i) = upper_bounds(j) - exceed;                       
		end       
   end
end
elseif kk == 2
for i = 1:m
	for j=1:n
        if ui_y(j,i)< lower_bounds(j)                                     
		    ui_y(j,i) =  lower_bounds(j);                     
		elseif ui_y(j,i) >  upper_bounds(j)                               
		    ui_y(j,i) = upper_bounds(j);                        
        end       
    end
end    
end

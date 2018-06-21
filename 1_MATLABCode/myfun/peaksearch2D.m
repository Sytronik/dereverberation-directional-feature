function [est_pos,location0] = peaksearch2D(hLocalMax,beampw,max_beampw,Np_phi,Np_th) 

        %% Peak search algorithm

            beampw_double = repmat(beampw,1,2); % replicate the matrix of beampw
            
            release(hLocalMax);
            hLocalMax.Threshold = max_beampw - 25;

            location0 = double(step(hLocalMax, beampw_double(1:end,:))); % convert uint32 index to double 
            
            location(:,1) = location0(:,1)/Np_phi*2*pi;
            location(:,2) = location0(:,2)/Np_th*pi;
            
            location = location(find(ge(location(:,1),pi).*lt(location(:,1),pi*3)),:); 
            location(ge(location(:,1),2*pi),1) = location(ge(location(:,1),2*pi),1)-2*pi;

            est_pos = [location(1,1) location(1,2);...
                            location(2,1) location(2,2)];
                

end
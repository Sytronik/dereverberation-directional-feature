azi_deg = 0:5:360;
azi = azi_deg * pi / 180;
r = 0:0.05:1;
[AZI, R] = meshgrid(azi, r)
[X, Y]=pol2cart(AZI, R);

color = zeros(21,73, 3);
color(:,:,1) = X;
color(:,:,2) = Y;

Z = sqrt(sum(color.^2,3));
color = (color+1)/2;

figure(1);clf;
polaraxes;
rticks(0);
thetaticks(0:15:360);
hold on;
title('Azimuth (deg) vs Color');
axes;
surf(X,Y, Z, color, 'edgecolor','none');
view(2);
axis 'equal' 'tight';
axis off;
hold off;
% xticks(0:30:360);
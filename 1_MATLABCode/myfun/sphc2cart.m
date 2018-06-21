function [x,y,z] = sphc2cart(az,elev,r)
%SPH2CART Transform spherical to Cartesian coordinates.
%   [X,Y,Z] = SPH2CART(TH,PHI,R) transforms corresponding elements of
%   data stored in spherical coordinates (azimuth TH, elevation PHI,
%   radius R) to Cartesian coordinates X,Y,Z.  The arrays TH, PHI, and
%   R must be the same size (or any of them can be scalar).  TH and
%   PHI must be in radians.
%
%   TH is the counterclockwise angle in the xy plane measured from the
%   positive x axis.  PHI is the elevation angle from the xy plane.
%
%   Class support for inputs TH,PHI,R:
%      float: double, single
%
%   See also CART2SPH, CART2POL, POL2CART.

%   Copyright 1984-2005 The MathWorks, Inc. 
%   $Revision: 5.12.4.2 $  $Date: 2005/05/31 16:31:22 $

z = r .* cos(elev);
rcoselev = r .* sin(elev);
x = rcoselev .* cos(az);
y = rcoselev .* sin(az);

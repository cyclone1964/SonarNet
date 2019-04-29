%computeSurfaceScatteringStrength - compute surface scatteringstrength
%
% Strength = 
%    computeSurfaceScatteringStrength(Type, Frequency, Angle,
%    Environment) 
%
% computes the surface scattering strength for the given frequency and
% angle. It supports multiple frequencies and angles: different
% frequencies are returned in different columnns, different angles in
% different rows.
%
% This is currently implmented as a table lookup, and presumes the
% existence of data files named 'SurfaceScatteringStrengthXX.dat', where
% 'XX' is the wind speed in m/s, and
% 'BottomScatteringStrengthXX.dat' where XXX is the bulk grain
% size, a number between -9 and 9. These parameters are taken from
% the environment and rounded to the nearest integer
%
% Copywrite 2010 BBN Technologies, Matt Daily author
function Strength = ...
    computeSurfaceScatteringStrength(Frequency, Angle, Environment)

persistent CurrentWindSpeed StrengthTable

Frequencies = (10000:1000:50000)';
Angles = 1:90;

Angle = max(1,min(90,Angle(:)' * 180/pi));
Frequency = Frequency(:);

% Check that the wind speed we have is the right wind speed
WindSpeed = ceil(Environment.Surface.WindSpeed);
if (isempty(CurrentWindSpeed) | WindSpeed ~= CurrentWindSpeed)
  % Load the table for the given grain size
  CurrentWindSpeed = WindSpeed;
  StrengthTable = load(['SRFSTR' num2str(WindSpeed,'%02d') '.dat']);
  StrengthTable = reshape(StrengthTable(:,2), ...
			  length(Angles), ...
			  length(Frequencies))';
end

% Now, do the interpolation
Strength = interp2(Angles,Frequencies,StrengthTable, Angle,Frequency);

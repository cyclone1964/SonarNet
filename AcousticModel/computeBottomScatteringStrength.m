%computeBottomScatteringStrength - compute bottom scattering strength
%
% Strength = 
%    computeBottomScatteringStrength(Frequency, Angle, Environment) 
%
% computes the bottom scattering strength for the given frequency and
% angle. It supports multiple frequencies and angles: different
% frequencies are returned in different columnns, different angles in
% different rows.
%
% This is currently implemnted as a table lookup, and presumes the
% existence of data files named 'BTMSTRXX.dat', where
% 'XX' is the Bulk Grain Size.
%
% Copywrite 2010 BBN Technologies, Matt Daily author
function Strength = ...
    computeBottomScatteringStrength(Frequency, Angle, Environment)

persistent CurrentGrainSize StrengthTable

Frequencies = (10000:1000:50000)';
Angles = 1:90;

Angle = max(1,min(90,Angle(:)' * 180/pi));
Frequency = Frequency(:);

% Check that the wind speed we have is the right wind speed
GrainSize = round(Environment.Bottom.GrainSize(1));
if (isempty(CurrentGrainSize) | GrainSize ~= CurrentGrainSize)
  % Load the table for the given grain size
  CurrentGrainSize = GrainSize;
  StrengthTable = load(['BTMSTR' num2str(GrainSize,'%+1d') '.dat']);
  StrengthTable = reshape(StrengthTable(:,2), ...
			  length(Angles), ...
			  length(Frequencies))';
end

% Now, do the interpolation
Strength = interp2(Angles,Frequencies,StrengthTable, Angle,Frequency);

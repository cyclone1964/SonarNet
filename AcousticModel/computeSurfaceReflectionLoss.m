%computeSurfaceReflectionLoss - compute surface scattering strength
%
% Loss = computeSurfaceReflectionLoss(Environment, Frequencies, Angles)
%
% computes the surface reflection loss for the given frequencies and
% angles. It supports multiple frequencies and angles: different
% frequencies are returned in different columnns, different angles in
% different rows.
%
% Copywrite 2010 BBN Technologies, Matt Daily author
function ReflectionLoss = ...
    computeSurfaceReflectionLoss(Frequencies, Angles, Environment)

% Shape the inputs appropriately
Angles = Angles(:);
Frequencies = Frequencies(:)';

% Get the windspeed
WindSpeed = Environment.Surface.WindSpeed;

% Compute the decay constant (m) and (km).
DecayDepthMeters = 0.07 * WindSpeed;
DecayDepthKilometers = 1.0e-3 * DecayDepthMeters;
       
% Compute the wind speed factor.
if (WindSpeed < 6)
  WindSpeedFactor = 6^1.57 * exp( 1.2 * (WindSpeed-6) ) / DecayDepthMeters;
else
  WindSpeedFactor = WindSpeed^1.57 / DecayDepthMeters;
end

% Compute the depth component integral.
Attenuation = ...
    integrateBubbleAttenuation(Environment, DecayDepthKilometers, Angles);
Attenuation = Attenuation * 1000;

% Compute the Frequency Factor
FrequencyFactor = 0.63e-3 * WindSpeedFactor * (Frequencies/1000).^0.85;

% Compute the surface bubble loss (dB)
ReflectionLoss = -min(Attenuation*FrequencyFactor,15);

%integrateBubbleAttenuation - Compute bubble integration
%
% Attenuation = integrateBubbleAttenuation(Environment, Decay, Angles)
% integrates the bubble attenuation for some reason or another.
function Attenuation = ...
    integrateBubbleAttenuation(Environment, Decay, Angles)

% The number of exponentials and intervals in there
NumIntervals = 16;
NumExponentials = 16;
NumDepths = NumExponentials * NumIntervals;

% Compute the exponentials: I have no idea what this means, but it
% runs from 1 to e^-16
Arg = -((1:NumDepths)-1)/NumIntervals;
Exponentials = exp(Arg);

% We presume the surface depth is 0 here
SurfaceDepth = 0;

% Compute a depth vector for the integration of attenuation
DepthIncrement = Decay/NumIntervals;
Depths = SurfaceDepth + ((1:NumDepths)-1) * DepthIncrement;

% Compute the sound speed at each depth.
SoundSpeeds = interp1(Environment.WaterColumn.Depths, ...
                      Environment.WaterColumn.SoundSpeeds, ...
                      Depths);
Slowness = cos(Angles) / SoundSpeeds(1);

% This forms NumAngles x NumDepths matrices
CosVector = Slowness * SoundSpeeds;
SinSqr = max(1 - CosVector.^2,0);
SinVector = sqrt(SinSqr);

% Do the integration, which is a sum of the 
Numerator = Exponentials(1:end-1) + Exponentials(2:end);
Numerator = repmat(Numerator,length(Angles),1);
Denominator = SinVector(:,1:end-1) + SinVector(:,2:end);
Numerator(Denominator == 0) = 0;
Denominator(Denominator == 0) = 1;
Integration = sum(Numerator ./ Denominator,2);

Attenuation = 2 * Integration * DepthIncrement;

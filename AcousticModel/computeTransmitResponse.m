%computeTransmitResponse - compute a receive beam pattern
%
% Beams = computeTransmitResponse(DirCos) computes a transmit beam
% response for a phase shaded square array. It is a four lambda
% array, and since the assumption is that the elements are half
% wavelength spaced, this is an 8 element array
function Beams = computeTransmitResponse(Directions,Steering)

% These are used to hold the tables, which different in the X and Y
% direction. 
persistent XResponse YResponse Offset Scale NumPoints

% If they are not defined, then load those tables
if (isempty(XResponse))

    % This makes a table of beam responses. It being a regular
    % array this is circular.
    NumPoints = 256; NumElements = 8;
    XResponse = zeros(NumPoints,1);
    Window = taylorwin(NumElements,5,-35)';
    Window = kaiser(NumElements)';
    
    % Now, the delay is a function distance from the center of the array
    Ramp = (1:NumElements); Ramp = sqrt((Ramp - mean(Ramp)).^2);
    Phase = (pi/4)*Ramp;
    Phase = Phase - mean(Phase);
    XResponse(1:NumElements) = (cos(Phase)+1i*sin(Phase)) .* Window;

    XResponse = abs(fft(XResponse));
    XResponse = XResponse/max(XResponse);

    Phase = (pi/4)*Ramp;
    Phase = Phase - mean(Phase);
    YResponse = zeros(NumPoints,1);
    YResponse(1:NumElements) = (cos(Phase)+1i*sin(Phase)) .* Window;

    YResponse = abs(fft(YResponse));
    YResponse = YResponse/max(YResponse);

    Scale = 0.5*NumPoints;
end

% Set a default steering
if (nargin < 2)
    Steering = zeros(3,1);
end

% Compute the response by computing the X and Y response and then
% adding the baffling.
Indices = mod(round(Scale*(Directions(1,:)'-Steering(1))),NumPoints)+1;
Beams = XResponse(Indices);
Indices = mod(round(Scale*(Directions(2,:)'-Steering(2))),NumPoints)+1;
Beams = Beams .* YResponse(Indices);
Beams = Beams .* computeBaffling(Directions);

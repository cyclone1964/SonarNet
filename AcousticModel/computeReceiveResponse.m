%computeReceiveResponse - compute a receive beam pattern
%
% Beams = computeReceiveBeamPattern(Directions) computes a receive beam
% response for a full-aperture kaiser shaded array with a 4
% wavelength aperture. 
%
% The underlying assumption is that the elements are half
% wavelength apart, and so there are 8 elements in a 4 lambda array

function Beams = computeReceiveResponse(Directions,Steering)

% These are variables we initialize once to make the table which we
% then index into to compute responses
persistent Response Offset Scale NumPoints

% Initialize those as necessary
if (isempty(Response))

    % This makes a table of beam responses. It being a regular
    % array this is circular.
    NumPoints = 256; NumElements = 8;
    Response = zeros(NumPoints,1);
    Response(1:NumElements) = taylorwin(NumElements,4,-35);
    Response = abs(fft(Response));
    Response = Response/max(Response);
    
    Scale = 0.5*NumPoints;
end

% If no steering is supplied, 
if (nargin < 2)
    Steering = [0 0 0]';
end

% Compute the response by computing the X and Y response and then
% adding the baffling.
Indices = mod(round(Scale*(Directions(1,:)'-Steering(1))),NumPoints)+1;
Beams = Response(Indices);
Indices = mod(round(Scale*(Directions(2,:)'-Steering(2))),NumPoints)+1;
Beams = Beams .* Response(Indices);
Beams = Beams .* computeBaffling(Directions);


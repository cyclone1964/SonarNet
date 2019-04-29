%genrateMMWindow  Generate a Mitchell-MacPhereson window
%
% Window = generateMMWindow(NumPoints) generates a
% Mitchell-MacPheresone window of length N. N must be a multiple of
% 2, and in reality this routine generates two windows: the front
% and back halves, each of length NumPoints. These are used in MM
% filtering to generate non-stationary sequences.
function Window = generateMMWindow(NumPoints)

Ramp = 0:(NumPoints/2);
Window.Front = sqrt(0.5 * (2 * Ramp'/NumPoints).^3);
Ramp = (NumPoints/2 + 1):(NumPoints-1);
Window.Front = [Window.Front
		sqrt(1 - 0.5 * (2 * (1 - Ramp'/NumPoints)).^3)];
Window.Back = Window.Front(end:-1:1);


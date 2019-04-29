%initializeSurfaceBoundary  initialize a surface boundary
%
% Surface = initializeSurfaceBoundary({PropertyList}) creates a
% new surface boundary. It sets defaults for the different surface
% boundary properties and then allows the user to override them.
%
% Copywrite 2010 BBN Technologies, Matt Daily author
function Surface = initializeSurfaceBoundary(varargin)

Surface.SeaState = 3;
Surface.WindSpeed = 5;
Surface.WaveHeight = 3;

Surface = setProperties(Surface, varargin{:});

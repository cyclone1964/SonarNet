%initializeEnvironment  initialize an environment structure
%
% Environment = initializeEnvironment({PropertyList}) initializes
% an environment. An environment consists of a BottomGrid, a
% SurfaceGrid, and a WaterColumn
%
% Copywrite 2010 BBN Technologies, Matt Daily author
function Environment = initializeEnvironment(varargin)

Environment.Surface = initializeSurfaceBoundary;
Environment.Bottom = initializeBottomBoundary;
Environment.WaterColumn = initializeWaterColumn;

Environment = setProperties(Environment,varargin{:});

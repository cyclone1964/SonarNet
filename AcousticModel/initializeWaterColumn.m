%initializeWaterColumn  initialize the water column (volume scattering profile)
%
% WaterColumn = initializeWaterColumn({PropertyList}) initializes
% the water column. Right now, it just sets a default volume
% scattering strength and sound speed
%
% Copywrite 2010 BBN Technologies, Matt Daily author
function WaterColumn = initializeWaterColumn(varargin)

WaterColumn.Depths = [0 1000]';
WaterColumn.SoundSpeeds = [1500 1500]';
WaterColumn.ScatteringStrength = -75;

WaterColumn = setProperties(WaterColumn, varargin{:});
%initializeBottomBoundary  initialize a BottomBoundary 
%
% Bottom = initializeBottomBoundary({PropertyList}) initializes a bottom
% boundary, specifically it's depth and bottom parameters.
%
% This model set includes two different models. One is a scattering
% model, the other a reflection model. The scattering model Roger
% Gauss's BSS model, implemented in computeBottomScattering. The
% second is the APL/UW 9407 reflection model, implemented in
% computeBottomReflection. 
%
% These two models operate on different sets of parameters. 
%
% The reflection model parameter is termed the "Bulk Grain Size" and
% has values according to the following table (more or less)
%
% -9.0 - Boulder
% -7.0 - Rock
% -3.0 - Gravel
%  1.5 - Sand
%  6.0 - Silt
%  7.0 - Mud
%  9.0 - Clay
%
% The scattering model operates on a "Bottom Type" which has the
% following equivalencies
%
% 1 - Basalt
% 2 - Rock
% 3 - Cobble
% 4 - Sandy Gravel
% 5 - Coarse Sand
% 6 - Medium Sand
% 7 - Fine Sand
% 8 - Silt

% Copywrite 2010 BBN Technologies, Matt Daily author
function Bottom = initializeBottomBoundary(varargin)

Bottom.GrainSize = 1.5;
Bottom.BottomType = 6; 
Bottom = setProperties(Bottom,varargin{:});


  
    
    
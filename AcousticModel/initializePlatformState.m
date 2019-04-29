%initializePlatformState  initialize a platform structure
%
% Platform = initializePlatformState creates a standard PlatformState
% structure with default properties set. The standard properties
% for a platform are given by:
%
% Position - location in feet
% Velocity - velocity of platform
% Attitude - Current Yaw/Pitch/Roll
% Orientation - Direction cosine representation of Attitude
%
% Copywrite 2010 BBN Technologies, Matt Daily author
function Platform = initializePlatformState(varargin)

Platform.Time = 0.0;
Platform.Position = [0 0 0]';
Platform.Velocity = [0 0 0]';
Platform.Attitude = [0 0 0]';
Platform.TurnRate = [0 0 0]';

Platform = setProperties(Platform,varargin{:});

% Now reshape the input vectors so that they are column vectors in
% accordance with the standards.
Platform.Position = Platform.Position(:);
Platform.Velocity = Platform.Velocity(:);
Platform.Attitude = Platform.Attitude(:);




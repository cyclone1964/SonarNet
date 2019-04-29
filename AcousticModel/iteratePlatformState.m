%iteratePlatformState  iterate a platform forward in time
%
% PlatformState = iteratePlatformState(PlatformState, Time)
% iterates the platform state forward in time.
function PlatformState = iteratePlatformState(PlatformState, Time)

while (Time < PlatformState.Time)
  Delta = min(0.1,Time - PlatformState.Time);
  PlatformState = turnPlatformState(PlatformState,Platform.TurnRate * Delta);
  PlatformState.Position = PlatformState.Position + ...
      Delta * PlatformState.Velocity; 
  Time = Time + Delta;
end



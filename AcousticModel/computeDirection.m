%computeDirection  Converts an attitude to an Direction
%
% Direction = computeDirection(Attitude) converts an attitude
% to an direction. Each column of the input is presumed to be a
% different attitude column vector. These inputs can have 1 or 2
% rows: missing rows will be presumed to be 0.
%
% Copyright 2006 BBN Technologies, Matthew Daily Author
function Direction = computeDirection(Attitude)

% If no input, return a 3 x empty matrix.
if (isempty(Attitude))
  Direction = zeros(3,0);
  return;
end

% Compute the direction
Direction = [cos(Attitude(2,:)) .* cos(Attitude(3,:))
	     cos(Attitude(2,:)) .* sin(Attitude(3,:))
	     sin(Attitude(2,:))];
%computeBaffling - compute the baffling of the array
%
% Baffling= computeBaffling(DirCos,Baffling) will return the
% backplane baffling of the array
%
function Baffling = computeBaffling(Direction,Baffling)

if (nargin < 2)
    Baffling = 10;
end

% Now attempt to put a backplane baffling, starting 
% 0.1 in front of the plane
Argument = min(0,Direction(3,:)'-0.1);
Baffling = exp(Baffling * Argument);


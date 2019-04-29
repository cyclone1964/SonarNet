%computeAttitude - convert a direction to an attitude
%
% Attitude = computeAttitude(Direction) computes the attitude from a
% direction cosine triplet. As a refresher an attitude is a vector of
% rotations about the X, Y, and Z axes that correspond to the given
% direction. The first is always 0 since a pointing direction does not
% provide information about rotation about the X axis.
%
% Copywrite 2010 BBN Technologies, Matt Daily author
function Attitude = computeAttitude(Direction)

Attitude = [0
	    -asin(Direction(3)/norm(Direction));
	    atan2(Direction(2),Direction(1))];

%computeRotationMatrix  compute a rotation matrix
%
% Matrix = computeRotationMatrix(Attitude) computes the rotation
% matrix for a given attitude vector.
%
% The Pitch or Roll members can be left off
%
function Matrix = computeRotationMatrix(Attitude)


% Parse the inputs according to the protocol defined above
Matrix = eye(3);
if (nargin < 1)
  return;
end


% Apply the Yaw Rotation First
Temp = [cos(Attitude(3)) -sin(Attitude(3)) 0
	sin(Attitude(3))  cos(Attitude(3)) 0
	0        0 1];
Matrix = Temp * Matrix;

% Then the pitch
Temp = [ cos(Attitude(2)) 0 sin(Attitude(2))
	 0 1          0
	 -sin(Attitude(2)) 0 cos(Attitude(2))];
Matrix = Temp * Matrix;
    
% Lastly the roll
Temp = [1          0          0
	0  cos(Attitude(1)) -sin(Attitude(1))
	0  sin(Attitude(1))  cos(Attitude(1))];
Matrix = Temp * Matrix;

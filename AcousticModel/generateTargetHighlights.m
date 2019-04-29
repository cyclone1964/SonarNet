%generateTargetHighlights - make a list of highlights
%
% Highlights = generateTargetHighlights(Source, Target) returns a
% series of Highlight structures with an entry for each
% highlight. This function models the target as a circular cylinder
% 100 meters long with 5 meter radius and two hemispherical caps plus
% a screw with some noise and a tower 1/3 of the way aft.
%
% The source and target inputs are Platform structures
function Highlights = generateTargetHighlights(Source,Target)

% Now, we set up the structure of the boat. Origin is under the sail,
% which is 1/3 down the way of the tube which is 100 meters long. It
% has two hemispherical caps on each end and a set of ribs.
Radius = 5;
FrontCapPosition = [33 0 0]';
EndCapPosition = [-67 0 0]';
ScrewPosition = [-77 0 0];
RibXPositions = (0:-4:-100) + 33;

% First, compute the global offset of the source in the target
% reference frame.
Offset = Source.Position - Target.Position;
Range = norm(Offset);
Direction = Offset/Range;
Speed = norm(Target.Velocity);

% Convert them to ship relative
Offset = computeRotationMatrix(Target.Attitude)' * Offset;
Direction = computeRotationMatrix(Target.Attitude)' * Direction;

% Now, to compute the specular, we break the area into three
% regions: forward, broadside, and aft. If we are within 5 degrees
% of broadside, we put the specular on the side of the boat and
% make it loud. Otherwise we put it on the bow or stern
% hemispheres. 
Limit = sind(5);
if (abs(Direction(1)) < sind(5))
    X = 33+100*(Direction(1) - Limit)/Limit;
    YZ = Radius * (Direction(2:3)/norm(Direction(2:3)));
    SpecularPosition = [X; YZ];
    
elseif (Direction(1) > 0)
    SpecularPosition = Radius * Direction + FrontCapPosition;
    
else
    SpecularPosition = Radius * Direction + EndCapPosition;
end

Highlights = newHighlight('Position',SpecularPosition, ...
                          'Strength', 10, ...
                          'Doppler',Speed*Direction(1));

% Now, we add the ribs. However, we only do this if the source is
% outside the frame of the sub.
if (norm(Offset(2:3)) > Radius) 
    Speed = norm(Target.Velocity);
    YZ = Radius * (Direction(2:3)/norm(Direction(2:3)));
    YZ = repmat(YZ,1,length(RibXPositions));
    RibPositions = [RibXPositions; YZ];
    for Index = 1:length(RibXPositions)
        Highlights(end+1) = newHighlight('Position',RibPositions(:,Index), ...
                                         'Strength', -4, ...
                                         'Doppler',Speed*Direction(1));
    end
end

% Now add some at the rear where the screw would be. This is done
% pretty stupidly: We presume that at 20 knots the screw turns 200
% RPM based upon a line from Hunt For Red October.
if (Speed > 100)
    RPM = 200*Speed/20;
    RPS = RPM/60;

    % Now, we generate random scatterers centered at the screw.
    YZ = randn(2,200);
    X = ScrewPosition(1)-Radius - exprnd(10,1,200);
    Positions = [X; YZ];
    
    % Now, the Doppler is a function of the radius of the scatterer
    R = sqrt(sum(YZ.^2,1));
    V = 2 * pi * R/RPS;
    Doppler = V .* randn(1,200);
    
    for Index = 1:length(Doppler)
        Highlights(end+1) = newHighlight('Position',Positions(:,Index), ...
                                         'Strength',-15-10*randn(1)^2, ...
                                         'Doppler',Doppler(Index));
    end
end

% Now, the positions are in target-relative coordinates: we have to
% go through and convert them to global.
for Index = 1:length(Highlights)
    Highlights(Index).False = false;
    Highlights(Index).Position = Target.Position + ...
        computeRotationMatrix(Target.Attitude)' * Highlights(Index).Position;
end

% Let's shape it into a column vector
Highlights = Highlights(:);

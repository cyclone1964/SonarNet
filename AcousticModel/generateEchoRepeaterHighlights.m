%generateEchoRepeaterHighlights - make a list of highlights
%
% Highlights = generateTargetHighlights(Source, Target) returns a
% series of Highlight structures with an entry for each
% highlight. This function models the target as a circular cylinder
% 100 meters long with 5 meter radius and two hemispherical caps plus
% a screw with some noise and a tower 1/3 of the way aft.
%
% The source and target inputs are Platform structures
function Highlights = generateEchoRepeaterHighlights(Source,Target)

% First, compute the ship-relative position of the source in the
% target reference frame.
Offset = Source.Position - Target.Position;
Range = norm(Offset);
Direction = Offset/Range;

% Now, the highlights occur in a line along the LOS to the source
% but behind us every 5 meters. 
Positions = Target.Position - Direction * (5:5:100);

Doppler = -10 + 20 * rand(1);

Highlights = newHighlight('Doppler',Doppler, ...
                          'Strength',10);
Highlights = repmat(Highlights,size(Positions,2),1);
for Index = 1:length(Highlights)
    Highlights(Index).False = true;
    Highlights(Index).Position = Positions(:,Index);
end

% Let's shape it into a column vector
Highlights = Highlights(:);

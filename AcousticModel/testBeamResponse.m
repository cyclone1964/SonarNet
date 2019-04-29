Array = initializeArray('Type','Circular');
Beam = initializeBeam('Array',Array,'Frequency',20000,'SoundSpeed',1500);

close all;

Directions = -1:0.01:1;
NumDirections = length(Directions);
Y = repmat(Directions,length(Directions),1);
Z = Y';
X = sqrt(max(0,1 - Z.^2 - Y.^2));

Directions = [X(:) Y(:) Z(:)]';

figure;
Response = computeElementResponse(Beam,Directions);
imagesc(reshape(abs(Response),NumDirections,NumDirections));
axis equal; axis tight;
title('Element Response at Initialized Frequency');

Response = computeBeamResponse(Beam,Directions);
figure;
imagesc(reshape(abs(Response),NumDirections,NumDirections));
axis equal; axis tight;
title('Beam Response at Initialized Frequency');


Response = computeBeamResponse(Beam,Directions,'Frequency',15000);
figure;
imagesc(reshape(abs(Response),NumDirections,NumDirections));
axis equal; axis tight;
title('Beam Response at Lower Frequency');

Response = computeBeamResponse(Beam,Directions,'Bearing',pi/4);
figure;
imagesc(reshape(abs(Response),NumDirections,NumDirections));
axis equal; axis tight;
title('Beam Response Steered to +X');


Response = computeBeamResponse(Beam,Directions,'DE',pi/4);
figure;
imagesc(reshape(abs(Response),NumDirections,NumDirections));
axis equal; axis tight;
title('Beam Response Steered to -Z');

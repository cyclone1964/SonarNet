% This script makes the pictures that I put onto my poster. These
% are representative samples of the data that I generate for
% training etc. 

% First the two platforms: the source, the target, and the decoy
Source = initializePlatformState('Position',[0 0 200], ...
                                 'Velocity',[10 0 0]);
Target = initializePlatformState('Position',[500*cosd(18) 500*sind(18) 200], ...
                                 'Attitude',[0 0 3*pi/4], ...
                                 'Velocity',[10 0 0]);
Decoy = initializePlatformState('Position',[1000,0,200], ...
                                'Attitude',[0 0 3*pi/4]);

Environment = initializeEnvironment;
Environment.Surface.WindSpeed = 10;
Environment.Bottom.GrainSize = 6;

% To make a picture of the target, generate highlights just for
% that and then render the target as a submarine and 
TargetHighlights = generateTargetHighlights(Source,Target);
useNamedFigure('Target'); clf; hold on;
renderSubmarine(Target); set(gca,'ZDir','reverse');
view(45,45);
Positions = [TargetHighlights.Position];
plot3(Positions(1,:)+10,Positions(2,:)+10,Positions(3,:),'k.','MarkerSize',20);
axis equal;
set(gca,'XTick',[],'YTick',[])
set(gca,'color','c');
LightHandle = light(gca,'Position',[400 -10 200]);

% Now make a picture of a decoy
useNamedFigure('Decoy'); clf; hold on;
DecoyHighlights = generateEchoRepeaterHighlights(Source, Decoy);
renderTorpedo(Decoy);
view(45,45);
Positions = [DecoyHighlights.Position];
plot3(Positions(1,:),Positions(2,:),Positions(3,:),'k.','MarkerSize',20);
axis equal;
set(gca,'XTick',[],'YTick',[])
set(gca,'color',[0.8 0.8 0.8]);
LightHandle = light(gca,'Position',[400 -10 200]);

% Now let's make a range dopler map and plot that.
Steerings = ...
    [zeros(1,25)
     repmat(-18,1,5) repmat(-9,1,5) zeros(1,5) repmat(9,1,5) repmat(18,1,5)
     repmat(-18:9:18,1,5)];
ReceiveDirections = computeDirection(Steerings * pi/180);

Highlights = [TargetHighlights; DecoyHighlights];
[Beams, Properties] = generateSamples('Highlights',Highlights, ...
                                      'PlatformState',Source, ...
                                      'ReceiveSteerings',ReceiveDirections, ...
                                      'Environment',Environment, ...
                                      'VolumeReverbAdjustment',-20, ...
                                      'BoundaryReverbAdjustment',[-20 -20]);
useNamedFigure('Spectrogram'); clf;
[S,F,T,P] = spectrogram(Beams(:,13),128,64,[],diff(Properties.Band));
F = F  - mean(F) + mean(Properties.Band);
imagesc(T*mean(Environment.WaterColumn.SoundSpeeds)/2,F,...
  10*log10(fftshift(P,1))); 
xlabel('Range (m)'); ylabel('Frequency'); title('Range Doppler Map');
prettyPlot;print('-dpng','RangeDoppler.png');
%hold on; plot([Properties.Pulses.Time],[Properties.Pulses.Frequency],'k.');



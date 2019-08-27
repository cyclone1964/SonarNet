% This script uses the functions developed to generate a lot of
% training data for my investigation into using Deep Learning to do
% the detection process. The training data consists of:
%
% Cycles - sonar cycles. These have 25 image planes, each a
% different interaction with either a target, a false target, or
% nothing. As currently structured, these are 256x32x25 images with
% values between 0 and 255 corresponding to waterfalls.
%
% LabelMaps - a single 256x32 image that is either 1 or 0 depending
% upon presence of the target at that location
%
% FeatureMaps - a file stating where the target is relative to
% the sonar and what the Doppler (this is an X/Y/Z/Doppler/valid
% fivetuple)

function makeTrainingData(path, num_samples)

% Set up the receive beams, steered every 9 degrees
Steerings = ...
    [zeros(1,25)
     repmat(-18,1,5) repmat(-9,1,5) zeros(1,5) repmat(9,1,5) repmat(18,1,5)
     repmat(-18:9:18,1,5)];
ReceiveDirections = computeDirection(Steerings * pi/180);

% Let's make a lot of these
SampleIndex = 0;
% while (SampleIndex < 40000)
while (SampleIndex < num_samples)
    
    % The environment has these things randomized:
    % WaterDepth, WindSpeed, and GrainSize
    while(true)
        GrainSize = round(-9+18*rand(1));
        if (GrainSize ~= 0)
            break;
        end
    end
    WindSpeed = 20*rand(1);
    WaterDepth = 250 + 1250*rand(1);
    Scattering = -60 - 20*rand(1);
    
    % The source is always at the origin, but the 
    Surface = initializeSurfaceBoundary('WindSpeed',WindSpeed);
    Bottom = initializeBottomBoundary('GrainSize',GrainSize);
    WaterColumn = ...
        initializeWaterColumn('Depths',[0 WaterDepth]', ...
                              'ScatteringStrength',Scattering);
    Environment = initializeEnvironment('WaterColumn',WaterColumn, ...
                                        'Surface',Surface, ...
                                        'Bottom',Bottom);
    
    % Now, the source is between the surface and the bottom, and
    % has a speed between 10 and 30 MPS
    SourceSpeed = 15;
    SourceDepth = 50 + (WaterDepth-100)*rand(1);
    
    Source = initializePlatformState('Position',[0 0 SourceDepth]', ...
                                     'Attitude',[0 0 0]', ...
                                     'Velocity',[SourceSpeed 0 0]');

    % Now choose a real target .. not every time though.
    if (rand(1) > 0)
        
        % The target is between 100 and 1300 meters away, and we
        % place it relatively to the source to keep it within the
        % visible window
        TargetRange = 50 + 700*rand(1);
        TargetBearing = -18 + 36 * rand(1);
        TargetElevation = -18 + 36 * rand(1);
        TargetPosition = Source.Position + ...
            TargetRange * computeDirection([0 TargetElevation TargetBearing]'*pi/180);

        % The target has random attitude in the X/Y plane and
        % random velocity up to 15 m/s (30 knots)
        TargetAttitude = [0 0 2*pi*rand(1)]';
        TargetVelocity = [10 * rand(1) 0 0];
        Target = initializePlatformState('Position',TargetPosition, ...
                                         'Attitude',TargetAttitude, ...
                                         'Velocity',TargetVelocity);

        TargetHighlights = generateTargetHighlights(Source,Target);
    else 
        
        % When we have none, we set these empty so we can tell below
        Target = [];
        TargetHighlights = [];
    end
    
    % And also make a false target sometimes
    if (rand() > 2)
        
        % The target is between 100 and 1300 meters away
        DecoyRange = 100 + 1400*rand(1);
        DecoyBearing = -18 + 36 * rand(1);
        DecoyElevation = -18 + 36 * rand(1);
        DecoyPosition = Source.Position + ...
            DecoyRange * computeDirection([0 DecoyElevation DecoyBearing]'*pi/180);
        
        DecoyAttitude = [0 0 2*pi*rand(1)]';
        DecoyVelocity = [15 * rand(1) 0 0];
        Decoy = initializePlatformState('Position',DecoyPosition, ...
                                         'Attitude',DecoyAttitude, ...
                                         'Velocity',DecoyVelocity);

        DecoyHighlights = generateEchoRepeaterHighlights(Source,Decoy);
    else 
        Decoy = [];
        DecoyHighlights = [];
    end

    % Generate the data 
    [Beams, Properties] =  ...
        generateSamples('Band',[19000, 21000], ...
                        'CycleLength',1, ...
                        'VolumeReverbAdjustment',-40, ...
                        'BoundaryReverbAdjustment',[0 0]', ...
                        'PlatformState',Source, ...
                        'Highlights',[TargetHighlights; DecoyHighlights], ...
                        'ReceiveSteerings',ReceiveDirections, ...
                        'Environment',Environment);

    % Now write the images into a file uniquely named from the time
    Id = mod(round(24*60*60*100*now),1000000000);
    FileName = sprintf(strcat(path, 'ImageMap-', int2str(Id), '.dat'));
    FID = fopen(FileName,'w');
    
    % Now do all the processing to generate the images for each or
    % the beams and write them into the image file
    for BeamIndex = 1:size(Beams,2)
        [~,F,T,P] = ...
            spectrogram(Beams(:,BeamIndex),64,32,64, ...
                        diff(Properties.Band));
        P(:,end+1) = P(:,end); T(end+1) = T(end)+diff(T(1:2));
        F = F  - mean(F) + mean(Properties.Band);
        P = 10 * log10(fftshift(P,1));
        P = max(0,min(255,P));
        fwrite(FID,P(:),'uint8');
    end
    fclose(FID);
    
    % Now make per bin labels
    Labels = zeros(size(P));
    if (~isempty(Properties.Pulses))
        Pulses = [Properties.Pulses];
        Bins = interp1(F,1:length(F),[Pulses.Frequency], ...
                       'nearest', 'extrap');
        Frames = interp1(T,1:length(T),[Pulses.Time], ...
                         'nearest','extrap');
        for Index = 1:length(Bins)
            if (Pulses(Index).False)
                Labels(Bins(Index),Frames(Index)) = 1;
            else
                Labels(Bins(Index),Frames(Index)) = 2;
            end
        end
    end
    
    % Write that to a file
    FileName = sprintf(strcat(path, 'LabelMap-', int2str(Id), '.dat'));
    FID = fopen(FileName,'w');
    fwrite(FID,Labels(:),'uint8');
    fclose(FID);
    fprintf('(%d): Generated Data Id %.0f\n',SampleIndex,Id);

    % Now make detection stats. These are either 0 for non or 1 for
    % true target. We remove decoys for this purpose
    BinFlags = zeros(size(F)); RangeFlags = zeros(size(T))';
    if (~isempty(TargetHighlights))
        Pulses = [Properties.Pulses];
        Pulses = Pulses(~[Pulses.False]);
        [~,Index] = max([Pulses.Strength]);
        Bin = interp1(F,1:length(F),Pulses(Index).Frequency, ...
                      'nearest', 'extrap');
        Frame = interp1(T,1:length(T),Pulses(Index).Time,  ...
                        'nearest','extrap');
        BinFlags(Bin) = 1;
        RangeFlags(Frame) = 1;

    end
    Detections = [BinFlags(:); RangeFlags(:)];

    % Write that to a file
    FileName = sprintf(strcat(path, 'Detections-', int2str(Id), '.dat'));
    FID = fopen(FileName,'w');
    fwrite(FID,Detections(:),'uint8');
    fclose(FID);

    % Now make a feature map. We allow room for up to 8 targets in
    % here, but we only populate the first one
    Features = zeros(40,1);
    Index = 0;
    if (~isempty(Target))
        Features(Index + (1:3)) = Target.Position;
        Features(Index + 4) = mean([TargetHighlights.Doppler]);
        Features(Index + 5) = 1;
        Index = Index + 5;
    end

    if (~isempty(Decoy))
        Features(Index + (1:3)) = mean([DecoyHighlights.Position],2);
        Features(Index + 4) = mean([DecoyHighlights.Doppler]);
        Features(Index + 5) = 2;
        Index = Index + 5;
    end
    
    FileName = sprintf(strcat(path, 'FeatureMap-', int2str(Id), '.dat'));
    FID = fopen(FileName,'w');
    fwrite(FID,Features(:),'float');
    fclose(FID);
    
    SampleIndex = SampleIndex + 1;
end

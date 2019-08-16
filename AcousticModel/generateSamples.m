%generateSamples - make a range doppler map with a target it in
%
% [Beams, Properties] = generateSamples(FileName,{PropertyList}) will
% generate a time series for a set of receive beams representing a
% complex baseband sonar signal. This is done from the following input
% properties.
%
% Band - band of sonar (2 x 1 Hz) [18000 22000]
% FrameSize - size of analysis FFT (dimensionless) (256)
% Environment - environment ot use
% SourceLevel - source level of emitter (dB) (200)
% PulseLength - length of pulse (s) (0.050)
% CycleLength - length of cycle (s) (2)
% PlatformState - State of the sonar platform
% BackgroundLevel - background noise level (dB/rootHz) (100)
% ReceiveSteerings - the steering directions of the receive beams (dircos)
% VolumeReverbLevel - scattering level of volumereverb (dB) (-60)
% BoundaryReverbAdjustment - dB adjustment for Surface (1) and Bottom (2) (0)
%
% This complex baseband sample rate is equal to the width of the given
% band, and the transmit pulse is at the center of the band ODN'd to
% account for platform motion so that the reverb ridge is in the
% middle. The sonar uses the default transmit beam and reeive beam
% shadings as defined in computeTransmitResponse and
% computeReceiveResponse.
function [Samples,Properties] = generateSamples(varargin)

% Set default properties and then modify for input list
Properties.Band = [18500 21500];
Properties.FrameSize = [];
Properties.Highlights = [];
Properties.Environment = initializeEnvironment;
Properties.SourceLevel = 220;
Properties.PulseLength = 0.050;
Properties.CycleLength = 2.75;
Properties.PlatformState = initializePlatformState;
Properties.BackgroundLevel = 80;
Properties.ReceiveSteerings = [0;0];
Properties.VolumeReverbAdjustment = -20;
Properties.BoundaryReverbAdjustment = [-150 -150]';

Properties = setProperties(Properties,varargin{:});

% Compute the sample rate and extract some
% commonly used platform and environment parametres.
CenterFreq = mean(Properties.Band);
SampleRate = diff(Properties.Band);
SoundSpeed = mean(Properties.Environment.WaterColumn.SoundSpeeds);
WaterDepth = Properties.Environment.WaterColumn.Depths(end);
PlatformDepth = Properties.PlatformState.Position(3);
PlatformSpeed = norm(Properties.PlatformState.Velocity);

% And the state of the platform
State = Properties.PlatformState;
NumBeams = size(Properties.ReceiveSteerings,2);

% If no frame size is given, let's estimate it from the length of the pulse
% and the sample rate;
if (isempty(Properties.FrameSize))
  Properties.FrameSize = ...
    2^(nextpow2(SampleRate * Properties.PulseLength)-1);
end
NumBins = Properties.FrameSize;

% From that, compute the bin width, frame length, etc
HzPerBin = SampleRate/NumBins;
SecPerFrame = NumBins/SampleRate;
SamplesPerFrame = NumBins;
NumFrames = ceil(Properties.CycleLength/SecPerFrame);
NumSamples = NumFrames * NumBins;

% ODN the transmit frequency
TransmitFreq = CenterFreq * (1-2*PlatformSpeed/SoundSpeed);

% In order to compute beampatterns, we need to rotate
% source-relative directions into array-relative directions (the
% beam pattern functions operate in a different coordinate system)
ArrayRotation = [0 -1 0; 0 0 -1; 1 0 0];

% Initialize with the noise
NoiseLevel = Properties.BackgroundLevel - 10 * log10(SampleRate);
Noise = 10^(NoiseLevel/20) * ...
        (randn(NumSamples,NumBeams) + 1i * randn(NumSamples,NumBeams));

% Initialize the signal
Signal = zeros(size(Noise));

% If there are any highlights defined, make signals for each or
% them. This we do in the time domain.
if (~isempty(Properties.Highlights))

    % Compute the offsets, ranges, and directions (in global
    % coordinates) of the target from the source.
    Offsets = [Properties.Highlights.Position] - ...
              repmat(State.Position, 1,length(Properties.Highlights));
    Ranges = sqrt(sum(Offsets.^2,1));
    Directions = Offsets ./ repmat(Ranges,3,1);
    
    % Now, we need to convert the directions into source-relative
    % coordinates
    Directions = computeRotationMatrix(State.Attitude) * Directions;
    
    % Now compute the Dopplers due to source velocity projected onto
    % those directions.
    SourceDopplers = State.Velocity' * Directions;
    
    % Now go through all the highlights and make the signals one by one
    for Index = 1:length(Ranges)

        % Compute the total doppler
        TotalDoppler = ...
            (1+2*SourceDopplers(Index)/SoundSpeed) * ...
            (1+2*Properties.Highlights(Index).Doppler/SoundSpeed);
        
        % We use the source level, prop loss, and target strength to compute a
        % level. Note that we sae this in a structure which we will
        % save in a vector below so that we can return this
        % information to the caller amended to the input
        % properties.
        Pulse.Strength = Properties.Highlights(Index).Strength;
        Pulse.Level = Properties.SourceLevel - ...
            40 * log10(Ranges(Index)) + ...
            Properties.Highlights(Index).Strength;

        % Compute the transmit beam response
        Transmit = ...
            computeTransmitResponse(ArrayRotation * Directions(:,Index));

        % Load the rest of the pulse parameters
        Pulse.False = Properties.Highlights(Index).False;
        Pulse.Level = Pulse.Level + 20*log10(Transmit);
        Pulse.Frequency = TotalDoppler * TransmitFreq;
        Pulse.Time = 2 * Ranges(Index)/SoundSpeed;

        % Now compute the doppler shifted length and baseband frequency of the
        % pulse
        NumPulseSamples = ...
            floor(Properties.PulseLength/TotalDoppler * SampleRate);
        BasebandFrequency = 2*pi*(Pulse.Frequency - CenterFreq)/SampleRate; 
        
        % Generate the actual pulse signals and window it
        Phase = (1:NumPulseSamples)*BasebandFrequency + 2*pi*rand(1);
        PulseSamples = (cos(Phase') + 1i * sin(Phase')) * 10^(Pulse.Level/20);
        PulseSamples = PulseSamples .* blackman(length(PulseSamples));
        
        % Add to the signal as much of it as fits in the cycle
        Indices = round(2 * Ranges(Index) * SampleRate/ SoundSpeed) + ...
                  (1:NumPulseSamples);
        Indices = Indices(Indices < NumSamples);
        PulseSamples = PulseSamples(1:length(Indices));
        
        % Scale each one by the receive beam and add it in
        for BeamIndex = 1:NumBeams 
            Receive = ...
                computeReceiveResponse(ArrayRotation*Directions(:,Index), ...
                                       ArrayRotation*...
                                       Properties.ReceiveSteerings(:,BeamIndex));
            Signal(Indices,BeamIndex) = Signal(Indices,BeamIndex) + ...
                Receive * PulseSamples;
        end
        
        % Now, we store the pulse parameters in the properties to
        % be returned as necessary.
        Properties.Pulses(Index) = Pulse;
    end
else
    % If no highlights, no pulses either!
    Properties.Pulses = [];
end


% Now let's make some volume reverb. To do this we need to integrate
% the beam pattern into a spectrum for each receive beam since, in
% this model, the volume reverb does not change spectrally as a
% function of time. THis is a full integration around a sphere.
AngleIncrement = pi/180;
Bearings = -(pi-AngleIncrement/2):AngleIncrement:(pi-AngleIncrement/2); 
Elevations = ...
    -(pi/2-AngleIncrement/2):AngleIncrement:(pi/2-AngleIncrement/2); 

% Now form integration points for all of them
[AllBearings,AllElevations] = meshgrid(Bearings,Elevations);

% Compute the transmit beam pattern by computing directions and then
% rotating into Array coordinates.
Directions = ArrayRotation * ...
    computeDirection([zeros(size(AllBearings(:))) ...
                    AllElevations(:) ...
                    AllBearings(:)]');
Transmit = computeTransmitResponse(Directions).^2;

% And the differential volume really
Differential = log10(SoundSpeed * Properties.PulseLength/2) * ...
    cos(AllElevations(:)) * AngleIncrement^2;

% Now compute the frequency bin for all of them based upon the
% projection of the platform speed onto the forward axis.
Temp = cos(AllElevations(:)) .* cos(AllBearings(:));
Freq = TransmitFreq*(1+2*PlatformSpeed*Temp/SoundSpeed);
Bins = round((Freq-Properties.Band(1))/HzPerBin);

% Need one for every receive beam. Now, some of these may be out of
% the frequency range of the sonar so we have to be careful of that.
Indices = find(Bins > 0);
Bins = Bins(Indices);
Transmit = Transmit(Indices);
Differential = Differential(Indices);
Directions = Directions(:,Indices);

VolumeSpectra = zeros(NumBins,NumBeams);
for BeamIndex = 1:NumBeams

    % Add receive pattern and integrate them into a spectrum
    Receive = ...
        computeReceiveResponse(Directions, ...
                               ArrayRotation * ...
                               Properties.ReceiveSteerings(:,BeamIndex)).^2;
    Spectrum = ...
        accumarray(Bins, ...
                   Receive .* ...
                   Transmit .* ...
                   Differential);
    if (length(Spectrum) < NumBins) 
        Spectrum(NumBins) = 0;
    end
    
    VolumeSpectra(:,BeamIndex) = fftshift(sqrt(Spectrum));
end

% This is a hack until I can get the real window
Window = generateMMWindow(NumBins);

% OK, having now done that, we make a series of frames of complex
% gaussian noise shaped by that
VolumeReverb = zeros(NumSamples,NumBeams);
LastFrame = zeros(NumBins,NumBeams);
SampleIndex = 1;
for FrameIndex = 1:NumFrames
    
    % We want every beam to have the same random noise component
    Temp = randn(NumBins,1) + 1i * randn(NumBins, 1);
    for BeamIndex = 1:NumBeams
        
        % Make the beam
        Frame = NumBins * sqrt(2) * ...
                ifft(VolumeSpectra(:,BeamIndex) .* Temp);
        
        % Add into the output for this beam at the proper indices
        Indices = (1:SamplesPerFrame) + SampleIndex-1;
        
        % On the first frame, we copy the frame directly, otherwise, we blend
        % this frame into the last one.
        if (FrameIndex > 1)
            VolumeReverb(Indices,BeamIndex) = ...
                Frame .* Window.Front + ...
                LastFrame(:,BeamIndex) .* Window.Back;
        else
            VolumeReverb(Indices,BeamIndex) = Frame;
        end

        % Save the last frame
        LastFrame(:,BeamIndex) = Frame;
    end
    
    % Next sample index
    SampleIndex = SampleIndex + NumBins;
end

% Now to first approximation let's just scale that by the two way
% prop loss to get the proper level.
VolumeTimes = ((1:size(VolumeReverb,1))-1)'/SampleRate + Properties.PulseLength;
Ranges = VolumeTimes * SoundSpeed/2;
PropagationLoss = 1 ./ Ranges;

% This level holds all the non-time dependent scaling factors
Level = Properties.SourceLevel + ...
        Properties.Environment.WaterColumn.ScatteringStrength + ...
        Properties.VolumeReverbAdjustment + ...
        10 * log10(HzPerBin);

% Convert to a scaling, add the prop loss, and scale
VolumeLevel = 10^(Level/20) * PropagationLoss;
VolumeReverb = VolumeReverb .* repmat(VolumeLevel,1,NumBeams);

% Now we have to make boundary reverb for the two boundaries.
% First, get times for the two boundaries where reverb actually hits. First, the time for each of the frames
FrameTimes = ((1:NumFrames)-1)*SecPerFrame + Properties.PulseLength;

% Now, load this with the boundary distances so we can use them in a loop
BoundaryDistances = [PlatformDepth;  WaterDepth - PlatformDepth];

% Angles for the integration (this one in cylindrical coordinates).
Angles = (-180:0.5:179)+0.5;
BoundaryReverb = zeros(NumSamples,NumBeams);

% For each of the two boundary types.
for BoundaryType = 1:2

  % Find those frames for this boundary that contribute to the signal
  FrameIndices = find(FrameTimes > ...
                      2*BoundaryDistances(BoundaryType)/SoundSpeed);
  
  % And now the times of those frames, both the front and the back.
  FrontTimes = FrameTimes(FrameIndices);
  BackTimes = FrontTimes + Properties.PulseLength;
  
  % These are the HORIZONTAL ranges, needed to form the scattering annuli
  FrontRanges = ...
      sqrt((FrontTimes*SoundSpeed/2).^2 - ...
           BoundaryDistances(BoundaryType)^2);
  BackRanges = ...
      sqrt((BackTimes*SoundSpeed/2).^2 - ...
           BoundaryDistances(BoundaryType)^2);
  
  % And the elevations to each frame
  Elevations = atan2d(BoundaryDistances(BoundaryType),FrontRanges);
  
  % The last frame for the MM windowing like before
  LastFrame = zeros(SamplesPerFrame,NumBeams);
  
  % Now do all the frames
  for FrameIndex = 1:length(FrameIndices)
    
    % Compute the directions to the scattering points
    Directions = ...
        computeDirection([zeros(size(Angles))
                        Elevations(FrameIndex) * ones(size(Angles))*pi/180
                        Angles*pi/180]);
    
    % Compute the frequency bins
    Freq = TransmitFreq * ...
                 (1+2*Directions(1,:) * PlatformSpeed/SoundSpeed);
    Bins = round((Freq - Properties.Band(1))/ HzPerBin);
    
    % Now compute the propagation loss, this time two way since we
    % don't gain the entire thing back in differential volume
    PropagationLoss = -40*log10(FrontTimes(FrameIndex)*SoundSpeed/2);
    Angle = abs(Elevations(FrameIndex)*pi/180);
    switch(BoundaryType)
      case 1
        Scattering = ...
            computeSurfaceScatteringStrength(CenterFreq, ...
                                             Angle, ...
                                             Properties.Environment);
      case 2
        Scattering = ...
            computeBottomScatteringStrength(CenterFreq, ...
                                            Angle, ...
                                            Properties.Environment);
    end
    
    % Compute the differential area: R dR dTheta
    DifferentialArea = FrontRanges(FrameIndex) * ...
        (BackRanges(FrameIndex) - FrontRanges(FrameIndex)) * ...
        min(diff(Angles)*pi/180);

    % Again, this has all the scalar stuff
    BoundaryLevel(FrameIndices(FrameIndex),BoundaryType) = ...
        Properties.SourceLevel + ...
        Scattering + ...
        PropagationLoss + ...
        Properties.BoundaryReverbAdjustment(BoundaryType) + ...
        10*log10(HzPerBin);
    BoundaryLevel(FrameIndices(FrameIndex),BoundaryType) = ...
        10^(BoundaryLevel(FrameIndices(FrameIndex),BoundaryType)/20);
    
    % Compute the transmit beam response
    Transmit = computeTransmitResponse(ArrayRotation * Directions).^2;

    % Remove bins out of range
    Indices = find(Bins > 0);
    Bins = Bins(Indices);
    Transmit = Transmit(Indices);
    Directions = Directions(:,Indices);

    % Make the noise the same for all of them
    Temp = randn(NumBins,1) + 1i * randn(NumBins,1);
    
    % Now for each beam ...
    for BeamIndex = 1:NumBeams

        % ... get the beam response and integrate it into a spectrum
        Receive = ...
            computeReceiveResponse(ArrayRotation*Directions, ...
                                   ArrayRotation * ...
                                   Properties.ReceiveSteerings(:,BeamIndex)).^2;
        Spectrum = sqrt(accumarray(Bins', ...
                                   Receive .* ...
                                   Transmit .* ...
                                   DifferentialArea));
        if (length(Spectrum) < NumBins) 
            Spectrum(NumBins) = 0;
        end

        % Now make the frame
        Spectrum = fftshift(Spectrum);
        Frame = NumBins * sqrt(2) * ifft(Spectrum .* Temp);
        Frame = Frame * BoundaryLevel(FrameIndices(FrameIndex),BoundaryType);
    
        % And place it into the reverberation. NOTE in this case, we don't do
        % the first frame thing since we want it to onset gently.
        Indices = (1:SamplesPerFrame) + ...
                  ((FrameIndices(FrameIndex)-1)*SamplesPerFrame);
        BoundaryReverb(Indices,BeamIndex) = ...
            Window.Back .* LastFrame(:,BeamIndex) + Window.Front .* Frame;
        LastFrame(:,BeamIndex) = Frame;
    end
  end
end

% Sum it all up
Samples = Noise + Signal + VolumeReverb + BoundaryReverb;
      

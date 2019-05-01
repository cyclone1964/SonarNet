%plotGeneratedData(ID) plots a given ID
%
% plotGeneratedData(ID) uses the input ID to retrieve the data from
% the generated data area and plots it
function plotGeneratedData(Ids,Beams)

% If no Ids are given, plot all of them
if (nargin < 1)
    Ids = load('../GeneratedData/Directory.txt');
end

% If no beams are given, max across all of them
if (nargin < 2)
    Beams = 1:25;
end

for Id = Ids(:)'

    % Load the image map
    FileName = sprintf('../GeneratedData/ImageMap-%d.dat',Id);
    FID = fopen(FileName,'rb');
    Image = fread(FID,[25*128,256],'uint8');
    Image = reshape(Image,256,128,25);
    fclose(FID);

    FileName = sprintf('../GeneratedData/LabelMap-%d.dat',Id);
    FID = fopen(FileName,'rb');
    LabelMap = fread(FID,[128*256],'uint8')';
    fclose(FID);
    LabelMap = reshape(LabelMap,256,128);

    FileName = sprintf('../GeneratedData/Detections-%d.dat',Id);
    FID = fopen(FileName,'rb');
    Detections = fread(FID,[128+256],'uint8')';
    fclose(FID);
    LabelMap = reshape(LabelMap,256,128);
    Bins = Detections(1:256);
    Frames = Detections(257:end);
    Bin = find(Bins); Frame = find(Frames);
    
    % Plot the watefall in the upper left 3/4 of the plot
    useNamedFigure('Image');
    subplot(4,4,[1,2,3 5,6,7,9,10,11]);
    imagesc(squeeze(max(Image(:,:,Beams),[],3)));
    set(gca,'XTick',[],'YTick',[]);
    title(sprintf('Id: %.0f (False: %.0f, True %.0f)', ...
      Id,length(find(LabelMap(:)==1)),...
      length(find(LabelMap(:)==2))));
    if (~isempty(Bin))
        hold on; plot(Frame,Bin,'ko','MarkerSize',10);
    end
    
    % And the image in the lower 1/4
    subplot(4,4,16);
    imagesc(LabelMap);
    set(gca,'XTick',[],'YTick',[]);
    
    % And the marginals
    subplot(4,4,[4,8,12]);
    Marginal = max(LabelMap,[],2);
    plot(Marginal,1:length(Marginal),'.');
    axis tight
    Temp = axis; Temp(1:2) = [0.5 1.5]; axis(Temp);
    set(gca,'XTick',[]); set(gca,'YTick',[]);
    set(gca,'XTick',[],'YTick',[]);
    
    subplot(4,4,[13,14,15]);
    Marginal = max(LabelMap,[],1);
    plot(1:length(Marginal),Marginal,'.');
    axis tight
    Temp = axis; Temp(3:4) = [0.5 1.5]; axis(Temp);
    set(gca,'XTick',[]); set(gca,'YTick',[]);
    
    fprintf('Key TO Continue\n');
    pause;

end
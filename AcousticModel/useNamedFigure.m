%useNamedFigure - set the figure to a named handle
%
% Handle = useNamedFigure(Name) returns the handle to the
% figure with the given name. If no such name has been used, it is created.
function Handle = useNamedFigure(Name)

persistent NamedFigures

Handle = [];
if (~isempty(NamedFigures))
  Index = find(strcmp(Name,NamedFigures(:,1)));
  if (~isempty(Index))
    Handle = NamedFigures{Index,2};
  end
end

if (isempty(Handle))
  Handle = figure;
  if (isempty(NamedFigures))
    NamedFigures = {Name Handle};
  else
    Index = size(NamedFigures,1)+1;
    NamedFigures{Index,1} = Name;
    NamedFigures{Index,2} = Handle;
  end
end

set(0,'CurrentFigure',Handle);

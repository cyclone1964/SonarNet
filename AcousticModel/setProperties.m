%setProperties  Set the fields in a structure 
%
% Struct = setProperties(Struct,'PropertyName',PropertyValue,Struct2)
% sets the fields in the structure Struct. For arguments that are
% character strings, it assumes that a property/value pair is
% provided. For arguments that are a struct, the ouput Structure
% "inherits" values from the argument. Any un-used property/value
% pairs are returned as an extra arguments
% 
% Copyright 2005 BBN Technologies, Matt Daily author
function [Struct, UnUsed] = setProperties(Struct, varargin)
  
ArgIndex = 1;
UnUsed = {};
while (ArgIndex <= length(varargin))
  if (ischar(varargin{ArgIndex}))
    if (ArgIndex == length(varargin))
      error 'Bad Property/Value Pair'
    end
    Property = varargin{ArgIndex};
    Value = varargin{ArgIndex+1};
    ArgIndex = ArgIndex + 2;
    
    if (isfield(Struct,Property))
      Struct = setfield(Struct,Property, Value);
      continue;
    else
      UnUsed = {UnUsed{:} Property Value};
      continue;
    end
  end
  
  if (isstruct(varargin{ArgIndex}))
    
    for FieldName =  fieldnames(varargin{ArgIndex})'
      FieldName = FieldName{1};
      Struct = setProperties(Struct, ...
			     FieldName, ...
			     getfield(varargin{ArgIndex},FieldName));
    end
    ArgIndex = ArgIndex + 1;
    continue;
  end
  
  error('Bad Argument');
end

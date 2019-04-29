function Hilight = newHighlight(varargin)

Hilight.Position = [0 0 0]';
Hilight.Strength = 0;
Hilight.Doppler = 0;

Hilight = setProperties(Hilight,varargin{:});

function prettyPlot(h)

% THE DEFAULTS
AxesFontName = 'Arial';
AxesFontSize = 11;
AxesFontWeight = 'bold';
AxesLineWidth = 2.5;

LineLineWidth = AxesLineWidth;
LineMarkerSize = 6;

RectangleLineWidth = AxesLineWidth;

TextFontName = AxesFontName;
TextFontWeight = AxesFontWeight;
TextFontSize = 14;

TitleFontName = AxesFontName;
TitleFontWeight = AxesFontWeight;
TitleFontSize = 16;

LabelFontName = AxesFontName;
LabelFontSize = 14;
LabelFontWeight = AxesFontWeight;


if (~exist('h'))
   h = gcf;
end;

switch get(h, 'Type')
   
   case 'axes',
      
      set(h, 'FontName', AxesFontName);
      set(h, 'FontSize', AxesFontSize);
      set(h, 'FontWeight', AxesFontWeight);
      set(h, 'LineWidth', AxesLineWidth);
      
      ht = get(h, 'Title');
      set(ht, 'FontName', TitleFontName);
      set(ht, 'FontWeight', TitleFontWeight);
      set(ht, 'FontSize', TitleFontSize);
      
      hlx = get(h, 'XLabel');
      set(hlx, 'FontName', LabelFontName);
      set(hlx, 'FontSize', LabelFontSize);
      set(hlx, 'FontWeight', LabelFontWeight);
      
      hly = get(h, 'YLabel');
      set(hly, 'FontName', LabelFontName);
      set(hly, 'FontSize', LabelFontSize);
      set(hly, 'FontWeight', LabelFontWeight);
      
      hlz = get(h, 'ZLabel');
      set(hlz, 'FontName', LabelFontName);
      set(hlz, 'FontSize', LabelFontSize);
      set(hlz, 'FontWeight', LabelFontWeight);
      
   case 'line',
      
      set(h, 'LineWidth', LineLineWidth);
      set(h, 'MarkerSize', LineMarkerSize);
      
   case 'rectangle',
      
      set(h, 'LineWidth', RectangleLineWidth);
      
   case 'text',
      
      set(h, 'FontName', TextFontName);
      set(h, 'FontWeight', TextFontWeight);
      set(h, 'FontSize', TextFontSize);
      
end;

Children = get(h, 'Children');
for ChildIndex = 1:length(Children),
   
   prettyPlot(Children(ChildIndex));
   
end
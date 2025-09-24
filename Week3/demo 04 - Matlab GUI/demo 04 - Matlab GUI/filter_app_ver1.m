function filter_app_ver1
% filter_app_ver1
% This version uses the Matlab "uifigure" to create the GUI. 
% The "uifigure" allows for the components (plots and controls)
% to be placed in the figure window using the "uigridlayout"
% function. 
%
% Ivan Selesnick. 2024

% Set up figure and controls

my_fig = uifigure;   % 'ui' means 'user interface'
my_fig.Name = "My Filter App";

movegui(my_fig, 'center');  % place figure window in center of screen

% Grid layout
grid = uigridlayout(my_fig, [3 1]);  
% 3 rows, 1 column, of components (plots, controls)

% Proportioning of components 
grid.RowHeight = {'1x', 'fit', 'fit'}; 
grid.ColumnWidth = {'1x'};
% '1x' means the size will stretch when the figure
% window is resized.

% Axis
ax = uiaxes(grid);

% Label
slider_label = uilabel(grid);
slider_label.Text = 'Cut-off frequency';

% Slider
slider = uislider(grid);
slider.Value = 0.3;
slider.Limits = [0 0.5];
slider.MajorTicks = 0:0.10:0.5;
slider.ValueChangingFcn = @slider_callback;   % callback function

N = 500;
n = 1:N;
x = sin(5*pi*n/N) + 0.5 * randn(1, N);        % input signal

% Specify that the lines, etc, go into the axis 'ax' defined above 
% (in the GUI).
line_handle = line(ax, n, x);
xlabel(ax, 'Time')
xlim(ax, [0, N]);
ylim(ax, [-3 3])

fc = slider.Value;
update_plot()

    function slider_callback(src, event)
        % callback function
        fc = event.Value;       % update value of fc
        update_plot()           % update plot
    end

    function update_plot()
        fc = max(fc, 0.01);
        fc = min(fc, 0.49);
        [b, a] = butter(2, 2*fc);  % Order-2 Butterworth filter (multiply fc by 2 for Matlab)
        y = filtfilt(b, a, x);
        set(line_handle, 'ydata',  y);        % Update data in figure
        title(ax, sprintf('Output of low-pass filter. Cut-off frequency = %.3f', fc));
    end

end

function Answer

% Set up figure and controls
my_fig = uifigure('Name',"Elliptic Filter");
movegui(my_fig,'center');

% grid layout
grid = uigridlayout(my_fig,[4 1]); 
grid.RowHeight = {'4x', 'fit', 'fit','fit'}; 
grid.ColumnWidth = {'1x'};

% sub-grid for plots
plots = uigridlayout(grid);
plots.RowHeight   = {'2x'};
plots.ColumnWidth = {'1x','1x'};

% Axes
ax1 = uiaxes(plots);
ax1.XGrid='on'; ax1.YGrid='on'; xlabel(ax1,'Normalized frequency'); 
ylabel(ax1,'|H(f)|'); title(ax1,'Frequency response'); 
xlim(ax1,[0 .5]); ylim(ax1,[0 1.2]); box(ax1,'on')

ax2 = uiaxes(plots);
ax2.XGrid='on'; ax2.YGrid='on'; xlabel(ax2,'Real Part'); 
ylabel(ax2,'Imaginary Part'); title(ax2,'Pole-Zero Plot'); 
box(ax2,'on')

% Sliders + labels
slider_fc_label = uilabel(grid); slider_fc_label.Text = 'Cut-off frequency';
slider_fc = uislider(grid); 
slider_fc.Value = 0.2; slider_fc.Limits = [0.01 0.49];
slider_fc.MajorTicks = 0:0.1:0.5; 

slider_rp_label = uilabel(grid); slider_rp_label.Text = 'Passband deviation';
slider_rp = uislider(grid);
slider_rp.Value = 0.02; slider_rp.Limits = [0.001 0.05];
slider_rp.MajorTicks = 0:0.01:0.05;

slider_rs_label = uilabel(grid); slider_rs_label.Text = 'Stopband deviation';
slider_rs = uislider(grid);
slider_rs.Value = 0.02; slider_rs.Limits = [0.001 0.05];
slider_rs.MajorTicks = 0:0.01:0.05;

% Lines for updating plots
freq_line = plot(ax1,nan,nan); hold(ax1,'on'); hold(ax1,'off');

% State variables
fc  = slider_fc.Value;
delp = slider_rp.Value;
dels = slider_rs.Value;
update_plot()

% Callbacks
slider_fc.ValueChangingFcn  = @(~,evt) (setval('fc',evt.Value));
slider_rp.ValueChangingFcn  = @(~,evt) (setval('delp',evt.Value));
slider_rs.ValueChangingFcn  = @(~,evt) (setval('dels',evt.Value));

    function setval(param,val)
        switch param
            case 'fc',   fc   = val;
            case 'delp', delp = val;
            case 'dels', dels = val;
        end
        update_plot()
    end

    function update_plot()
        % Convert deviations to dB
        Rp = -20*log10(1-delp);
        Rs = -20*log10(dels);

        % Elliptic filter design
        [b,a] = ellip(3,Rp,Rs,fc*2);

        % Frequency response
        [H,w] = freqz(b,a,1024,'half'); f = w/(2*pi);
        set(freq_line,'XData',f,'YData',abs(H));
        title(ax1,sprintf('Frequency response (fc=%.2f, Rp=%.3f, Rs=%.3f)',fc,delp,dels));

        % Pole-zero plot
        cla(ax2)
        zplane(b,a,ax2);
    end
end

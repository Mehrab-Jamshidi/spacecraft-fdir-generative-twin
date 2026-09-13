clc
close all

% Extract data from Simulink output
t      = out.tout;
w_true = squeeze(out.omega_true.signals.values)';
w_meas = squeeze(out.omega_meas.signals.values)';
f_flag = squeeze(out.fault_flag.signals.values)';
var_x  = squeeze(out.variance_x.signals.values)';
fdir   = squeeze(out.fdir_flag.signals.values)';

figure('Position', [100 100 1400 900])
sgtitle('Spacecraft Attitude Simulation — Detumbling with Gyroscope Drift Fault', ...
        'FontSize', 13, 'FontWeight', 'bold')

subplot(4,1,1)
plot(t, vecnorm(w_true, 2, 2)*180/pi, 'b', 'LineWidth', 1.2)
hold on
xregion(300, 600, 'FaceColor', [0.84 0.19 0.15], 'FaceAlpha', 0.08)
xline(300, '--r', 'LineWidth', 1.2)
ylabel('|\omega| [deg/s]'); grid on
title('Angular rate magnitude — detumbling converges then fault disrupts controller')
legend('True |\omega|', 'Fault active')

subplot(4,1,2)
plot(t, w_true(:,1)*180/pi, 'b', 'LineWidth', 1.0, 'DisplayName', 'True \omega_x')
hold on
plot(t, w_meas(:,1)*180/pi, 'Color', [0.99 0.55 0.35], ...
     'LineWidth', 0.8, 'DisplayName', 'Gyro measurement (noise + drift)')
xregion(300, 600, 'FaceColor', [0.84 0.19 0.15], 'FaceAlpha', 0.08)
xline(300, '--r', 'LineWidth', 1.2)
ylabel('\omega_x [deg/s]'); grid on; legend
title('Gyroscope x-axis — drift fault causes progressive divergence from true value')

subplot(4,1,3)
err = vecnorm(w_meas - w_true, 2, 2)*180/pi;
plot(t, err, 'r', 'LineWidth', 1.2)
hold on
xregion(300, 600, 'FaceColor', [0.84 0.19 0.15], 'FaceAlpha', 0.08)
xline(300, '--r', 'LineWidth', 1.2)
ylabel('Error [deg/s]'); grid on
title('Estimation error — grows continuously after fault onset')
annotation('textbox', [0.55 0.38 0.3 0.04], 'String', ...
    'Fault injected at t=300s', 'Color', 'r', 'FontSize', 8, 'EdgeColor', 'none')

subplot(4,1,4)
plot(t, var_x, 'Color', [0.10 0.59 0.25], 'LineWidth', 1.2, ...
     'DisplayName', 'Running variance \omega_x')
hold on
threshold = mean(var_x(t < 300)) * 5;
yline(threshold, ':', 'Color', [0.46 0.42 0.69], 'LineWidth', 1.5, ...
      'DisplayName', sprintf('Detection threshold = %.5f', threshold))
plot(t, fdir*threshold*1.2, 'k--', 'LineWidth', 1.0, 'DisplayName', 'FDIR flag raised')
xregion(300, 600, 'FaceColor', [0.84 0.19 0.15], 'FaceAlpha', 0.08)
xline(300, '--r', 'LineWidth', 1.2)
ylabel('Variance [rad^2/s^2]')
xlabel('Time [s]')
grid on; legend
title('Running variance (classical FDIR) — drift fault may evade threshold detection')

exportgraphics(gcf, 'simulink_spacecraft_simulation.png', 'Resolution', 150)
disp('Figure saved as simulink_spacecraft_simulation.png')
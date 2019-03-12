clear;
points_num = [20, 50, 100, 200, 300, 500, 1000, 5000];
cal_pi_mean = [];
cal_pi_var = [];

for num = points_num
    cal_pi = [];
    for x = 1:20
        in_count = 0;
        for y = 1:num
            x_cor = rand(1);
            y_cor = rand(1);
            if x_cor*x_cor + y_cor*y_cor <= 1
                in_count = in_count + 1;
            end
        end
        cal_pi = [cal_pi in_count * 4 / num];
    end
    cal_pi_mean = [cal_pi_mean mean(cal_pi)];
    cal_pi_var = [cal_pi_var var(cal_pi)];
end

% subplot(2,1,1);
plot(points_num,cal_pi_mean,'--');
title('Adopt the Monte Carlo method to estimate the value of pi');
xlabel('the number of points in the unit square');
ylabel('the mean of estimated value of pi');
% subplot(2,1,2);
% plot(points_num,cal_pi_var);
% xlabel('the number of points in the unit square');
% ylabel('the variance of estimated value of pi');
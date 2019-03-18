clear;
sample_num = [10, 20, 30, 40, 50, 60, 70, 80, 100, 200, 500];

cal_integrate_mean = [];
cal_integrate_var = [];

for num = sample_num
    cal_integrate = [];
    for x = 1:100
        cal_sum = 0;
        for y = 1:num
            x_cor = rand(1) * 2 + 2;
            y_cor = rand(1) * 2 - 1;
            cal_sum = cal_sum + 4 *(power(y_cor,2) * exp(-power(y_cor,2)) + power(x_cor,4) * exp(-power(x_cor,2))) / (x_cor * exp(-power(x_cor,2)));
        end
        cal_integrate = [cal_integrate cal_sum / num];
    end
    cal_integrate_mean = [cal_integrate_mean mean(cal_integrate)];
    cal_integrate_var = [cal_integrate_var var(cal_integrate)];
end

% fun = @(x,y) (y.^2 .* exp(-y.^2) + x.^4 .* exp(-x.^2)) / (x .* exp(-x.^2));
% q = integral2(fun,2,4,@(x)-1,@(x)1);
% subplot(2,1,1);
plot(sample_num,cal_integrate_mean);
title('Adopt the Monte Carlo method to calculate the integrate of f(x)');
xlabel('the number of samples');
ylabel('the mean value of estimated value of integrate');
% subplot(2,1,2);
% plot(sample_num,cal_integrate_var);
% xlabel('the number of samples');
% ylabel('the variance of estimated value of integrate');

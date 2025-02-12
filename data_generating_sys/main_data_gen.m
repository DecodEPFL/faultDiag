clear;close all;clc;

load('input_fault_training_data','u1_training','u2_training','f_training')

params;

y1_training   = [];
y2_training   = [];
y3_training   = [];
y4_training   = [];
u1_training_s = [];
u2_training_s = [];
for i=1:size(u1_training,1)
    t           = 0:T:t_final;
    input       = [t;u1_training(i,:);u2_training(i,:)]';
    out         = sim('sys_sim');
    t           = out.tout;
    x           = out.x;
    y_iter      = out.y;
    

    % plot(t,y_iter(:,1))
    % hold on
    % plot(t,f_training(i,:),'r')
    % 
    % figure
    % plot(t,y_iter(:,2))
    % hold on
    % plot(t,f_training(i,:),'r')

 
    
    y1_training   = [y1_training;y_iter(1:T_sampling/T:end,1)']; %%pick every 30th element to lower data length
    y2_training   = [y2_training;y_iter(1:T_sampling/T:end,2)'];
    y3_training   = [y3_training;y_iter(1:T_sampling/T:end,3)'];
    y4_training   = [y4_training;y_iter(1:T_sampling/T:end,4)'];
    u1_training_s = [u1_training_s;u1_training(i,1:T_sampling/T:end)]; 
    u2_training_s = [u2_training_s;u2_training(i,1:T_sampling/T:end)]; 
    close all
end

%%20 rows u1, 20 u2, etc. 
training_data = [u1_training_s;u2_training_s;y1_training;y2_training;y3_training;y4_training;f_training(:,1:T_sampling/T:end)];

save('training_data','training_data')

%% Test data
load('input_fault_test_data','u1_test','u2_test','f_test')

y1_test     = [];
y2_test     = [];
y3_test     = [];
y4_test     = [];
u1_test_s   = [];
u2_test_s   = [];

for i=1:size(u1_test,1)
    t           = 0:T:t_final;
    input       = [t;u1_test(i,:);u2_test(i,:)]';
    out         = sim('sys_sim');
    t           = out.tout;
    x           = out.x;
    y_iter      = out.y;
    
    
    y1_test = [y1_test;y_iter(1:T_sampling/T:end,1)'];
    y2_test = [y2_test;y_iter(1:T_sampling/T:end,2)'];
    y3_test = [y3_test;y_iter(1:T_sampling/T:end,3)'];
    y4_test = [y4_test;y_iter(1:T_sampling/T:end,4)'];
    u1_test_s = [u1_test_s;u1_test(i,1:T_sampling/T:end)]; 
    u2_test_s = [u2_test_s;u2_test(i,1:T_sampling/T:end)];
    close all
end

test_data = [u1_test_s;u2_test_s;y1_test;y2_test;y3_test;y4_test;f_test(:,1:T_sampling/T:end)];

save('test_data','test_data')

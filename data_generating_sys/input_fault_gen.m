%% Input and fault data generation and save
clc;clear;close all;

% Define the parameters
lower_bound_wu      = 0.3*2*pi;
upper_bound_wu      = 1.5*2*pi;
lower_bound_au      = 0.01; 
upper_bound_au      = 0.1;
lower_bound_phiu    = 0*pi/180;
upper_bound_phiu    = 170*pi/180;
lower_bound_nu      = 2;
upper_bound_nu      = 10;
lower_bound_au_f    = 0.01;     
upper_bound_au_f    = 0.1;     

training_data_num   = 20;

% Generate 1000+training_data_num random variables from the uniform distribution of variables in each of the above intervals
params;
t           = 0:T:t_final;
u1_dict     = zeros(1000+training_data_num,length(t));
u2_dict     = zeros(1000+training_data_num,length(t));
f_dict      = zeros(1000+training_data_num,length(t));
% dic_nu      = [];
% dic_wu      = [];
% dic_phiu    = [];
% dic_au      = [];
% dic_nut     = [];
% dic_wut     = [];
% dic_phiut   = [];
% dic_aut     = [];

for j =1:training_data_num+1000
    u1      = 0;
    u2      = 0;
    f       = 0;
    nu      = round(lower_bound_nu + (upper_bound_nu - lower_bound_nu) * rand(1));
    % if j<training_data_num+1
    %    dic_nu    = [dic_nu;nu];
    % else
    %    dic_nut    = [dic_nut;nu];
    % end
    sum_au   = [0;0];
    sum_au_f = 0;
    for i = 1:nu
        au      = lower_bound_au + (upper_bound_au - lower_bound_au) * rand(2,1);
        au_f    = lower_bound_au_f + (upper_bound_au_f - lower_bound_au_f) * rand(1);
        sum_au  = sum_au + au;
        sum_au_f= sum_au_f + au_f;
        wu      = lower_bound_wu + (upper_bound_wu - lower_bound_wu) * rand(3,1);
        phiu    = lower_bound_phiu + (upper_bound_phiu - lower_bound_phiu) *rand(3,1);
        u1      = u1+au(1)*sin(wu(1)*t+phiu(1));
        u2      = u2+au(2)*sin(wu(2)*t+phiu(2));
        f       = f+au_f*sin(wu(3)*t+phiu(3));
        % %% to save which frequencies are in training and test
        % if j<training_data_num+1
        %     dic_wu      = [dic_wu;wu];
        %     dic_au      = [dic_au;au];
        %     dic_phiu    = [dic_phiu;phiu];
        % else
        %     dic_wut     = [dic_wut;wu];
        %     dic_aut     = [dic_aut;au];
        %     dic_phiut   = [dic_phiut;phiu];
        % end
    end
    u1_dict(j,:) = u1/sum_au(1)*upper_bound_au;
    u2_dict(j,:) = u2/sum_au(2)*upper_bound_au;
    f_dict(j,:)  = f/sum_au_f*upper_bound_au_f;
end
%%no input at time zero
u1_dict(:,1) = 0;
u2_dict(:,1) = 0;
f_dict(:,1) = 0;
% plot(dic_au,dic_wu,'*')
%% save training and test data 
u1_training = u1_dict(1:training_data_num,:);
u2_training = u2_dict(1:training_data_num,:);
f_training  = f_dict(1:training_data_num,:);
u1_test     = u1_dict(training_data_num+1:end,:);
u2_test     = u2_dict(training_data_num+1:end,:);
f_test      = f_dict(training_data_num+1:end,:);
% save('training_data_details','dic_wu','dic_au','dic_phiu','dic_nu')
% save('test_data_details','dic_wut','dic_aut','dic_phiut','dic_nut')
save('input_fault_training_data','u1_training','u2_training','f_training')
save('input_fault_test_data','u1_test','u2_test','f_test')
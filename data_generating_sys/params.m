
t_final             = 20;
T                   = 1/100;%% simulation params

noise_activation    = 0;

T_sampling          = 1/4;

%% System Parameters
mass    = 580/1000;    
mt1     = 36.26/1000;  
mt2     = mt1;
In      = 63.3316/1000; 
len     = 1.524;
c1      = 710.70/1000/5;
c2      = c1;
k1      = 19357.2/1000;
k2      = k1;
k13     = 15000/1000;
k23     = k13;
kt1     = 96319.76/1000;
kt2     = kt1;


Ma       = [ mass/2  mass/2  0   0
           -In/len  In/len  0   0
              0       0    mt1  0
              0       0     0  mt2];

K       = [    k1         k2         -k1       -k2      zeros(1,4)
           -len/2*k1   len/2*k2    len/2*k1  -len/2*k2  zeros(1,4)
              -k1         0         k1+kt1       0      zeros(1,4)
               0         -k2          0       k2+kt2    zeros(1,4)];


A       = [ zeros(4,4)      eye(4)          
                   -inv(Ma)*K       ];

inp_cof = [ 0   0 
            0   0
           kt1  0
            0  kt2];

B_u     = [   zeros(4,2)
           inv(Ma)*inp_cof];                                          
    
C       = [1 0 -1  0 zeros(1,4)
           0 1  0 -1 zeros(1,4)
           zeros(1,4) 1 0 -1  0
           zeros(1,4) 0 1  0 -1];
  

S_eta_2 = -inv(Ma)*[1          1   
                -len/2      len/2
                  -1         0
                   0         -1];

S_eta    = [   zeros(4,1)  zeros(4,1)  
                      S_eta_2];                                 

%eta = [0.3*k1_act*vetax_1 + k13*vetax_1^3
%       0.3*k1_act*vetax_2 +k13*vetax_2^3]  %note that k2 = k1, k13= k23
%%or linear part of eta = (A_act-A)*x which is = S_eta*[0.3*k1_act*vetax_1
%                                                0.3*k1_act*vetax_1]

V_eta     = [1 0 -1 0 zeros(1,4)
             0 1  0 -1 zeros(1,4)];

S_g    = S_eta;


V_g   = [zeros(1,4) 10 0 -10  0
         zeros(1,4) 0 10  0 -10]; %% 10 is because f_c = c_i 0.2 *tanh(10 dq) so 10 is added to V_g                                          

%%adding linear c to A and A_act
A       = A+0.2*c1*S_g*V_g;

n = size(A,1); 
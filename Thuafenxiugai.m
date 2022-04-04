%@AUTHOR: Lumin Liu during the Internship at Huawei Tech. Company. 
%Please do not distribute.
%part of the codes are taken from the file "demo_model.m" in cost-2100 on github.
%%
clc;
clear all;

%choose a network
Network = 'IndoorHall_5GHz';

Link = 'Single';
Antenna = 'MIMO_VLA_omni';
Band = 'Wideband';

scenario = 'LOS'; % {'LOS'} only LOS is available
freq = [-10e6 10e6]+5.3e9; % [Hz}
snapRate = 25; % Number of snapshots per s
snapNum = 10; % Number of snapshots
%MSPos  = [10 5 0]; % [m]
MSVelo = [0.001 0 0]; % [m/s]
BSPosCenter  = [0 0 0]; % Center position of BS array [x, y, z] [m]
BSPosSpacing = [0.0283 0 0]; % Inter-position spacing (m), for large arrays
BSPosNum = 32; % Number of positions at each BS site, for large arrays

train = 75000
test = 25000;

%train set
HT = zeros(train, snapNum*2048);
i = 0;
row_i = 1;
while row_i <= train
	rng('default');
	rng(i);
    MSPos = [((randi(2,1,2)-1)*2-1).*(10*rand(1,2)), 0];%MSPos = [(randi(2,1,2)-1)*2-1.*(10*rand(1,2)), 0];
    try
		[...
		    paraEx,...       % External parameters
		    paraSt,...       % Stochastic parameters
		    link,...         % Simulated propagation data for all links [nBs,nMs]
		    env...           % Simulated environment (clusters, clusters' VRs, etc.)
		] = cost2100...
		(...
		    Network,...      % Model environment
		    scenario,...     % LOS or NLOS
		    freq,...         % [starting freq., ending freq.]
		    snapRate,...     % Number of snapshots per second
		    snapNum,...      % Total # of snapshots
		    BSPosCenter,...  % Center position of each BS
		    BSPosSpacing,... % Position spacing for each BS (parameter for physically very-large arrays)
		    BSPosNum,...     % Number of positions on each BS (parameter for physically very-large arrays)
		    MSPos,...        % Position of each MS
		    MSVelo...        % Velocity of MS movements
		    ); 
	catch
		warning('This random seed is not working.');
	end

	delta_f = (freq(2) - freq(1))/1023;
	h_omni_MIMO = create_IR_omni_MIMO_VLA(link,freq,delta_f,Band);
    h = squeeze(h_omni_MIMO);
	h1 = ifft(h, [], 3);
	h_concate = h1(:,1:32, :);
	h_concat = permute(h_concate, [1,3,2]);
    h_conc = [real(h_concat), imag(h_concat)];
    h_con = reshape(h_conc, [1,snapNum*2*32*32]);
	%normalize to [0,1] range
	HT(row_i, :) =  h_con;
	i = i + 1;
	row_i = row_i + 1;
end

save('DATA_HT10_trainin', 'HT');

%test set
HT = zeros(test, snapNum*2048);
i = 0;
row_i = 1;
while row_i <= test
	rng('default');
	rng(i);
    MSPos = [((randi(2,1,2)-1)*2-1).*(10*rand(1,2)), 0];%MSPos = [(randi(2,1,2)-1)*2-1.*(10*rand(1,2)), 0];
    try
		[...
		    paraEx,...       % External parameters
		    paraSt,...       % Stochastic parameters
		    link,...         % Simulated propagation data for all links [nBs,nMs]
		    env...           % Simulated environment (clusters, clusters' VRs, etc.)
		] = cost2100...
		(...
		    Network,...      % Model environment
		    scenario,...     % LOS or NLOS
		    freq,...         % [starting freq., ending freq.]
		    snapRate,...     % Number of snapshots per second
		    snapNum,...      % Total # of snapshots
		    BSPosCenter,...  % Center position of each BS
		    BSPosSpacing,... % Position spacing for each BS (parameter for physically very-large arrays)
		    BSPosNum,...     % Number of positions on each BS (parameter for physically very-large arrays)
		    MSPos,...        % Position of each MS
		    MSVelo...        % Velocity of MS movements
		    ); 
	catch
		warning('This random seed is not working.');
	end

	delta_f = (freq(2) - freq(1))/1023;
	h_omni_MIMO = create_IR_omni_MIMO_VLA(link,freq,delta_f,Band);
    h = squeeze(h_omni_MIMO);
	h1 = ifft(h, [], 3);
	h_concate = h1(:,1:32, :);
	h_concat = permute(h_concate, [1,3,2]);
    h_conc = [real(h_concat), imag(h_concat)];
    h_con = reshape(h_conc, [1,snapNum*2*32*32]);
	%normalize to [0,1] range
	HT(row_i, :) =  h_con;
	i = i + 1;
	row_i = row_i + 1;
end

save('DATA_HT10_testin', 'HT');
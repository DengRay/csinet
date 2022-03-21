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
snapRate = 1; % Number of snapshots per s
snapNum = 1; % Number of snapshots
%MSPos  = [10 5 0]; % [m]
%MSVelo = [0 0 0]; % [m/s]
BSPosCenter  = [0 0 0]; % Center position of BS array [x, y, z] [m]
BSPosSpacing = [0.0283 0 0]; % Inter-position spacing (m), for large arrays
BSPosNum = 32; % Number of positions at each BS site, for large arrays

train = 100000;
val = 30000;
test = 20000;

%train set
HT = zeros(train, 2048);
i = 0;
row_i = 1;
while row_i <= train
	rng('default');
	rng(i);
    MSPos = [((randi(2,1,2)-1)*2-1).*(10*rand(1,2)), 0];%MSPos = [(randi(2,1,2)-1)*2-1.*(10*rand(1,2)), 0];
    v=0.1;
    temp = v*rand(1);
    MSVelo = [temp,sqrt(v^2-temp^2),0];
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
	h = ifft(h, [], 2);
	h = h/max(abs(h(:)));%h = h/max(abs(h(:)))./2;
	h_concate = h(1:32, :);
	h_concat = permute(h_concate, [2,1]);
	h_con = reshape(h_concat, [1, 32*32]);
	%normalize to [0,1] range
	HT(row_i, :) = [real(h_con), imag(h_con)];
	i = i + 1;
	row_i = row_i + 1;
end

save('DATA_HTtrainin', 'HT');
%{
	h = squeeze(h_omni_MIMO);
	h1 = ifft(h, [], 2);
 	h_concate = h1(1:32, :);
    h_concat = permute(h_concate, [2,1]);
    temp=abs(h_concate(:));
    temp1=temp.^2;
    temp2=sum(sum(temp1));
    temp3=sqrt(temp2);
	h2 = h_concat/temp./2;
	h_con = reshape(h2, [1, 32*32]);
	%normalize to [0,1] range
	HT(row_i, :) = [real(h_con), imag(h_con)]+ 0.5;
    %}

%val set
HT = zeros(val, 2048);
i = 0;
row_i = 1;
while row_i <= val
	rng('default');
	rng(i);
    MSPos = [((randi(2,1,2)-1)*2-1).*(10*rand(1,2)), 0];%MSPos = [(randi(2,1,2)-1)*2-1.*(10*rand(1,2)), 0];
    v=0.1;
    temp = v*rand(1);
    MSVelo = [temp,sqrt(v^2-temp^2),0];
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
	h = ifft(h, [], 2);
	h = h/max(abs(h(:)));%h = h/max(abs(h(:)))./2;
	h_concate = h(1:32, :);
	h_concat = permute(h_concate, [2,1]);
	h_con = reshape(h_concat, [1, 32*32]);
	%normalize to [0,1] range
	HT(row_i, :) = [real(h_con), imag(h_con)];
	i = i + 1;
	row_i = row_i + 1;
end

save('DATA_HTvalin', 'HT');


%test set
HT = zeros(test, 2048);
i = 0;
row_i = 1;
while row_i <= test
	rng('default');
	rng(i);
    MSPos = [((randi(2,1,2)-1)*2-1).*(10*rand(1,2)), 0];;%MSPos = [(randi(2,1,2)-1)*2-1.*(10*rand(1,2)), 0];
    v=0.1;
    temp = v*rand(1);
    MSVelo = [temp,sqrt(v^2-temp^2),0];
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
	h = ifft(h, [], 2);
	h = h/max(abs(h(:)));%h = h/max(abs(h(:)))./2;
	h_concate = h(1:32, :);
	h_concat = permute(h_concate, [2,1]);
	h_con = reshape(h_concat, [1, 32*32]);
	%normalize to [0,1] range
	HT(row_i, :) = [real(h_con), imag(h_con)];
	i = i + 1;
	row_i = row_i + 1;
end

save('DATA_HTtestin', 'HT');

%Simulation-1

%define slices
slices = {'eMBB','URLLC','mMTC'};
bandwidth = [50e6,10e6,5e6]; %in Hz
latency = [20e-3,1e-3,50e-3]; %in seconds
dataRate = [100e6,1e6,500e3]; %in bps

%initialize resource grid for each slide with added noise for interference
resourceGrids = cell(1,length(slices));
noiseLevel = 0.05;
for i = 1:length(slices)
    resourceGrids{i} = struct('Bandwidth',bandwidth(i), ...
                              'Latency',latency(i), ...
                              'DataRate',dataRate(i), ...
                              'NoiseLevel',noiseLevel*randn(1,10)); %random noise for interference
end

%simulate traffic, interference and allocation
results = struct();
for i = 1:length(slices)
    traffic = generateTraffic(dataRate(i));
    allocatedResources = allocateResourcesWithInterference(resourceGrids{i},traffic);
    results.(slices{i}) = struct('Traffic',traffic,'Allocated',allocatedResources, ...
            'Interference', resourceGrids{i}.NoiseLevel.*traffic);
end

%plot the results for visualization
figure;
subplot(2,1,1);
hold on;
for i = 1:length(slices)
    plot(1:10, results.(slices{i}).Allocated,'-o','DisplayName',[slices{i} 'Allocated']);
end
hold off;
xlabel('Time Slot');
ylabel('Allocated Bandwidth (Hz)');
title('Network Slicing Bandwidth Allocation with Interference');
legend;
grid on;

subplot(2,1,2);
hold on;
for i = 1: length(slices)
    plot(1:10,results.(slices{i}).Interference, '--', 'DisplayName',[slices{i} 'Interference']);
end
hold off;
xlabel('Time Slot');
ylabel('Interference Level');
title('Interference Impact on Each Slice');
legend;
grid on;

% === Function Definations ===

%function to simulate complex traffic pattern
function traffic = generateTraffic(dataRate)
    timeSlots = 10;
    traffic = dataRate *(0.8+0.4*rand(1,timeSlots));
end

%function to allocate resources considering interference
function allocatedResources = allocateResourcesWithInterference(grid,traffic)
    interference = traffic.*grid.NoiseLevel;
    adjustedTraffic = traffic-interference;
    allocatedResources = adjustedTraffic.*(grid.Bandwidth/sum(adjustedTraffic));
end
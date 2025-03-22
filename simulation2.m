%define slices and priorities
slices = {'eMBB','URLLC','mMTC'};
bandwidth =  [50e6, 10e6, 5e6]; %in Hz
latency = [20e-3,1e-3,50e-3]; %in seconds
dataRate = [100e6,1e6,500e3]; %in bps
priorities = [1,3,2]; %higher value indicates higher priority (URLLC>mMTC>eMBB)

%resource grid for each slice
resourceGrids = cell(1,length(slices));
noiseLevel = 0.05;
for i = 1:length(slices)
    resourceGrids{i} = struct('Bandwidth',bandwidth(i), ...
                              'Latency', latency(i), ...
                              'DataRate', dataRate(i), ...
                              'NoiseLevel', noiseLevel*rand(1,10));
end

%simulate user mobility
cellCount = 3; %base station count
userPerSlice = randi([5,15],length(slices),cellCount); %3*3 matrix

%simulate traffic, interference and allocation
results = struct();
totalLatencyPenalities = zeros(1,length(slices));
for i = 1:length(slices)
    traffic = generateComplexTraffic(dataRate(i),sum(userPerSlice(i,:)));
    allocatedResources = allocateResourcesWithPriority(resourceGrids{i},traffic,priorities(i));
    latencyPenalty = calculateLatencyImpact(latency(i),traffic);
    totalLatencyPenalities(i) = latencyPenalty;
    results.(slices{i}) = struct('Traffic', traffic, ...
                                'Allocated',allocatedResources, ...
                                'Interference',resourceGrids{i}.NoiseLevel, ...
                                'LatencyPenalty',latencyPenalty);
end

%visualization
figure;

%plot allocated bandwidth
subplot(3,1,1);
hold on;
for i = 1:length(slices)
    plot(1:10,results.(slices{i}).Allocated, '-o','DisplayName',[slices{i} ' Allocated']);
end
hold off;
xlabel('Time slot');
ylabel('Allocated Bandwidth (Hz)');
title('Network Slicing Bandwidth Allocation with Priority');
legend;
grid on;

%plot interference impact
subplot(3,1,2);
hold on;
for i = 1:length(slices)
    plot(1:10,results.(slices{i}).Interference, '--','DisplayName',[slices{i} 'Interference']);
end
hold off;
xlabel('Time Slot');
ylabel('Interference Level');
title('Interference Impact on Each Slice');
legend;
grid on;

%plot Latency Penalty
subplot(3,1,3);
bar(categorical(slices),totalLatencyPenalities);
xlabel('Slice');
ylabel('Latency Penalty');
title('Latency Impact on Each Slice');
grid on;

% === Function definations ===

%function to simulate traffic
function traffic = generateComplexTraffic(dataRate, userCount)
    timeSlots = 10;
    traffic = dataRate*userCount*(0.8+0.4*rand(1,timeSlots));
end

%function to allocate resources considering priority and interference
function allocatedResources = allocateResourcesWithPriority(grid, traffic, priority)
    interference = traffic.*grid.NoiseLevel;
    adjustedTraffic = traffic-interference;
    priorityFactor = priority/sum(priority); %normalize priority
    allocatedResources = adjustedTraffic.*(grid.Bandwidth/sum(adjustedTraffic))*priorityFactor;
end

%function to simulate latency impact
function latencyPenalty = calculateLatencyImpact(sliceLatency, traffic)
    delayImpact = (traffic>sliceLatency).*(traffic-sliceLatency);
    latencyPenalty = sum(delayImpact);
end
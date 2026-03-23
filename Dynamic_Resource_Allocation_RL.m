% Scenario 2: Dynamic Resource Management using RL
clear; clc; close all;

% System Parameters
nUEs = 4;           % Number of users
nRBs = 8;           % Available Resource Blocks
nStates = 10;       % Network states
nActions = nRBs;    % Actions (specific for each user)

% Q-Learning Parameters
alpha = 0.1;        % Learning Rate
gamma = 0.9;        % Discount Factor
epsilon = 0.2;      % Exploration
nEpisodes = 500;    % How many times the system will be trained

% Initializing Q-Table (States x Actions)
Q = zeros(nStates, nActions);

% Training the RL Agent
fprintf('Έναρξη εκπαίδευσης Reinforcement Learning...\n');
reward_history = zeros(nEpisodes, 1);
for ep = 1:nEpisodes
   state = randi(nStates); % Starting from a random state
   total_reward = 0;
  
   for step = 1:20 % 20 steps
       % Selecting action
       if rand < epsilon
           action = randi(nActions); % Exploration
       else
           [~, action] = max(Q(state, :)); % Choosing the best option
       end
     
       % If the RB is clean => high reward, if it has interference => low reward
       interference = rand * 5; % Random interference in the specific RB
       throughput = 10 * log2(1 + (20 / (1 + interference))); % Shannon formula
      
       reward = throughput; % The reward is the speed we achieved
      
       % Next State
       next_state = randi(nStates);
      
       % Updating Q-Table (Bellman Equation)
       Q(state, action) = Q(state, action) + alpha * (reward + gamma * max(Q(next_state, :)) - Q(state, action));
      
       state = next_state;
       total_reward = total_reward + reward;
   end
   reward_history(ep) = total_reward;
end

% Comparing and evaluating
% Comparing with a random resource
random_rewards = 50 + randn(nEpisodes, 1) * 10;

% Showing Results
figure;
plot(smoothdata(reward_history), 'LineWidth', 2, 'Color', 'b'); hold on;
plot(smoothdata(random_rewards), 'LineWidth', 1.5, 'Color', 'r', 'LineStyle', '--');
grid on;
xlabel('Επεισόδια Εκπαίδευσης (Episodes)');
ylabel('Συνολική Απόδοση (Throughput / Reward)');
legend('RL-Based Resource Allocation', 'Random Allocation (Baseline)');
title('Σύγκριση Απόδοσης MAC Layer: AI vs Traditional');

% To try Dynamic Resource with fairness, just uncomment this code block and
% comment the previous one 
%{ 
Scenario 2_1: Dynamic Resource Management using RL - Comparing max
% throuput, resource fair and proportional fair
clear; clc; close all;
% System Parameters
nUEs = 4;           % Number of users
nRBs = 8;           % Available Resource Blocks
nStates = 10;       % Network states
nActions = nUEs;    % Actions (specific for each user)
%adding episodes
nEpisodes = 500;    % Episodes
% Q-Learning Parameters
alpha = 0.1;        % Learning Rate
gamma = 0.9;        % Discount Factor
epsilon = 0.15;      % Exploration
%scenarios
scenarios = {'Max Throughput', 'Resource Fair', 'Proportional Fair'};
all_results = zeros(nEpisodes, 3);
all_fairness = zeros(nEpisodes, 3);
for scen = 1:3
   % Initializing Q-Table (States x Actions)
   Q = zeros(nStates, nActions);
  
   avg_throughput = ones(1, nUEs) * 5; %for proportional fair
   times_picked_total = zeros(1, nUEs); % for Resource Fair
   for ep = 1:nEpisodes
   state = randi(nStates); % Starting from a random state
   total_ep_throughput = zeros(1, nUEs);
  
   for step = 1:20 % 20 steps
       % Selecting action
       if rand < epsilon
           action = randi(nActions); % Exploration
       else
           [~, action] = max(Q(state, :)); % Choosing the best option
       end
      
       user_idx = action;
       % If the RB is clean => high reward, if it has interference => low reward
       interference = rand * 5; % Random interference in the specific RB
       throughput = 10 * log2(1 + (20 / (1 + interference))); % Shannon formula
       if scen == 1 % Max Throughput
           reward = throughput; % The reward is the speed we achieved
       elseif scen == 2 % Resource Fair
           reward = 1 / (times_picked_total(user_idx) + 1); %not fair to have too many resources
       else % Proportional Fair
           reward = throughput / (avg_throughput(user_idx) + 1e-6);
       end
          
       % Next State
       next_state = randi(nStates);
      
       % Updating Q-Table (Bellman Equation)
       best_next_q = max(Q(next_state, :));
       Q(state, action) = Q(state, action) + alpha * (reward + gamma * max(Q(next_state, :)) - Q(state, action));
      
       total_ep_throughput(user_idx) = total_ep_throughput(user_idx) + throughput;
       times_picked_total(user_idx) = times_picked_total(user_idx) + 1;
       avg_throughput(user_idx) = 0.9 * avg_throughput(user_idx) + 0.1 * throughput;
       state = next_state;
   end
  
       %calculating Jain's Fairness Index: (sum(x)^2) / (n * sum(x^2))
       sum_tp = sum(total_ep_throughput);
       sum_sq_tp = sum(total_ep_throughput.^2);
       if sum_sq_tp > 0
       jains_index = (sum_tp^2) / (nUEs * sum_sq_tp);
       else
           jains_index = 0;
       end
      
       all_results(ep, scen) = sum_tp;
       all_fairness(ep, scen) = jains_index;
   end
end
%results
%plot 1 => trhouput
figure('Position', [100, 100, 1200, 500], 'Color', 'w');
subplot(1,2,1);
plot(smoothdata(all_results, 1, 'movmean', 50), 'LineWidth', 2);
title('Total Network Throughput');
xlabel('Episodes'); ylabel('Total Mbps per Episode');
legend(scenarios, 'Location', 'best'); grid on;
%plot 2 => Fairness Index
subplot(1,2,2);
plot(smoothdata(all_fairness, 1, 'movmean', 50), 'LineWidth', 2);
ylim([0 1.1]);
title('Jain''s Fairness Index (Smoothed)');
xlabel('Episodes'); ylabel('Fairness (0 to 1)');
legend(scenarios, 'Location', 'best'); grid on;
%}
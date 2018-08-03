function [dataTraining,dataTesting,ns,datn] = prepare_data(data,timestamp_format,CaseID,Timestamp,Activity,x)
%PREPARE_DATA Summary of this function goes here
%   Detailed explanation goes here
%Convert Timestamp
data(:,Timestamp) = datetime(data(:,Timestamp),'TimeZone','Europe/London','InputFormat',timestamp_format);

data = sortrows(data,[CaseID,Timestamp]) %Make sure log is sorted by CaseID and Timestamp
data(:,Timestamp) = []; %Remove Timestamp Column as it is not needed anymore
%% 
%Make sure CaseID is first column and Activity second column
[datlen datn] = size(data);
ind = [1:datn];
ind = ind(find(ind~=CaseID));
ind = ind(find(ind~=Activity));
data = data(:,[CaseID, Activity, ind]); 

% Delete columns with more than 30 different values
[datlen datn] = size(data);
del_index = [];
for i=3:datn
    k = length(unique(data(:,i)));
    if k>30
        del_index = [del_index; i];
    end
end
data(:,del_index) = [];

%Allow max q columns, delete rest
q = 4;
[datlen datn] = size(data);
if datn>q
    data(:,[q+1:datn]) = [];
end

%Get info from data
[datlen datn] = size(data); %datlen = number of rows, datn = number of columns

% Determine dimensions for each attribute in log
unique_attr = cell(1,datn);
unique_values = cell(1,datn);
for i=1:datn
    unique_attr{i} = sort(unique(data(:,i)));
    unique_values{i} = length(unique(data(:,i)));
end

% Convert log to numbers (faster and necessary for algorithm)
data_num = ones(datlen,datn);
mapping = cell(1,datn);
for i=1:datn
    mapping{i} = cell(1,unique_values{i});
    mapping{i} = cellstr(unique_attr{i});
    %mapping{i}(1,:) = num2cell(1:unique_values{i});
    data_num(:,i) = double(categorical(data(:,i)));
end
save('mapping_table.mat','mapping');

ncases = unique_values{1}; % get number of cases from log
Q = 4; % num hidden states %input from user
unique_values{1} = Q; %replace cases count by number of states
ns = cell2mat(unique_values);%number of states

% Split data by case, remove case id and save in cell array
 [~,~,X] = unique(data_num(:,CaseID));
 data_num(:,CaseID) =  string(missing); %remove CaseID and create empty values for hidden state
 data_cell = accumarray(X,1:size(data_num,1),[],@(r){data_num(r,:)});
 

p = x/100      % proportion of rows to select for training
N = size(data_cell,1)  % total number of rows 
tf = false(N,1)    % create logical index vector
tf(1:round(p*N)) = true     
tf = tf(randperm(N))   % randomise order
dataTraining = data_cell(tf,:) 
dataTesting = data_cell(~tf,:) 

end


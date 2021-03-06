function [dataTraining,dataTesting,unique_values,N, mapping] = prepare_data(filename, delimiter,timestamp_format,CaseID,Timestamp,Activity,x,split_stable, model, blow_up_train, blow_up_test, max_num_context)
%PREPARE_DATA 
% Load csv file, transform as necessary and return training and test data
% set
%  Input
%%      filename = path including filename of csv file ("./examples/data.csv")
%%      delimiter (e.g. "," or ";")
%%      timestamp_format = format of timestamp (e.g. "d.M.YYYY")
%%      CaseID = position of CaseID column (e.g. 1)
%%      Timestamp = position of Timestamp Column (e.g. 2)
%%      Activity = position of Activity column (e.g. 3)
%%      x = percentage of training set split size (e.g. "70")
%%      split_stable = produce same data and test set everytime or shuffle ("yes" or "no")
%%      model = model type (e.g. "hmm" or "dbn")
%%      blow_up_train = if to add additional cases to the data set for each partial trace of the learn log
%%      blow_up_test = if to add additional cases to the data set for each partial trace of the test log
%%      max_num_context = maximum number of context attributes considered (e.g 4)
%%
%   Output
%%      dataTraining = data set for training
%%      dataTesting = data set for testing
%%      unique_values = vector with number of unique values for each attribute (e.g. [100 4 6])
%%      N = number of attributes in data set
%%      mapping = mapping table

disp('Start Loading Data');

%Load Data
data = import_csv(filename, delimiter); 

%Convert Timestamp
data(:,Timestamp) = datetime(data(:,Timestamp),'TimeZone','Europe/London','Format',timestamp_format);

data = sortrows(data,[CaseID,Timestamp]); %Make sure log is sorted by CaseID and Timestamp
data(:,Timestamp) = []; %Remove Timestamp Column as it is not needed anymore
%% 
%Make sure CaseID is first column and Activity second column
[datlen datn] = size(data);
ind = [1:datn];
ind = ind(find(ind~=CaseID));
ind = ind(find(ind~=Activity));
data = data(:,[CaseID, Activity, ind]); 

% Delete columns with more than 100 different values or only 1 different
% value
[datlen datn] = size(data);
del_index = [];
for i=3:datn % start at because ignoring case id and activity
    k = length(unique(data(:,i)));
    if k>100 || k==1 % k 0 number of different values
        del_index = [del_index; i];
    end
end
data(:,del_index) = [];

%Allow max q columns, delete rest
[datlen datn] = size(data);
if datn>max_num_context + 2
    data(:,[max_num_context+3:datn]) = [];
end

%if hmm or pfa delete all columns but observation
if strcmp(model,'hmm') || strcmp(model,'pfa')
    data = data(:,1:2);
end

%Get info from data
[datlen datn] = size(data); %datlen = number of rows, datn = number of columns

% Determine dimensions for each attribute in log
unique_attr = cell(1,datn);
unique_values = cell(1,datn);
for i=1:datn
    unique_attr{i} = sort(unique(data(:,i))); %save unique values for each attribute for mapping
    unique_values{i} = length(unique(data(:,i))); %save number of different values for each attribute
end

% Convert log to numbers (faster and necessary for algorithm)
% Create mapping table to be able to map numbers to real values again
data_num = ones(datlen,datn);
mapping = cell(1,datn);
for i=1:datn
    mapping{i} = cell(1,unique_values{i});
    mapping{i} = cellstr(unique_attr{i});
    data_num(:,i) = double(categorical(data(:,i)));
end  

ncases = unique_values{1}; % get number of cases from log

% Split data by case, remove case id and save in cell array
 [~,~,X] = unique(data_num(:,CaseID));
 data_num(:,CaseID) =  string(missing); %remove CaseID and create empty values for hidden state
 data_cell = accumarray(X,1:size(data_num,1),[],@(r){data_num(r,:)});
 
% Delete cases with only one or two events as trace length has to be at least 3
del_ind = [];
for i=1:ncases
    if size(data_cell{i},1)<3
        del_ind = [del_ind;i];
    end
end
data_cell(del_ind) = [];

%Split in test and training data
p = x/100  ;    % proportion of rows to select for training
N = size(data_cell,1);  % total number of rows 
tf = false(N,1);    % create logical index vector
tf(1:round(p*N)) = true;    
if strcmp(split_stable,'yes')
    rng(0);
end
tf = tf(randperm(N));   % randomise order
dataTraining = data_cell(tf,:) ;
dataTesting = data_cell(~tf,:) ;

N = datn -1 + 1; % number of variables in one time slice. datn + 1 (for hidden state) -1 (for case id column)

%Blow up data sets by inserting each partial sequence of a case as a new case
ncasesTrain = size(dataTraining,1);
if strcmp(blow_up_train,'yes')
    nparticalcases = 0;
    for i=1:ncasesTrain
        T = size(dataTraining{i},1);
        nparticalcases = nparticalcases + T - 2; %-2 because minimum trace length is 3
    end
    data_blown_up = cell(nparticalcases,1);
    ind = 0;
    for i=1:ncasesTrain
        T = size(dataTraining{i},1);
        for j=3:T
            ind = ind + 1;
            data_blown_up{ind} = dataTraining{i}(1:j,:);
        end
    end
    dataTraining = data_blown_up;    
    ncasesTrain = nparticalcases;
end

ncasesTest = size(dataTesting,1);
if strcmp(blow_up_test,'yes')
    nparticalcases = 0;
    for i=1:ncasesTest
        T = size(dataTesting{i},1);
        nparticalcases = nparticalcases + T - 2; %-2 because minimum trace length is 3
    end
    data_blown_up = cell(nparticalcases,1);
    ind = 0;
    for i=1:ncasesTest
        T = size(dataTesting{i},1);
        for j=3:T
            ind = ind + 1;
            data_blown_up{ind} = dataTesting{i}(1:j,:);
        end
    end
    dataTesting = data_blown_up;    
    ncasesTest = nparticalcases;
end

unique_values{1} = ncasesTrain + ncasesTest;

disp('Finish Loading Data');
disp('');

end


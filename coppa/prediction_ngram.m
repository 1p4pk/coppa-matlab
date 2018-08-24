function [pred real_value] = prediction_ngram(dataTraining,dataTesting,unique_values)
%NGRAM_ACCURACY Summary of this function goes here
%   Detailed explanation goes here
%% N-Gram Benchmark
ncases = size(dataTraining,1);
traces = cell(1,ncases);
nevents = 0;
td = containers.Map;


for i=1:ncases
    T = size(dataTraining{i},2);
    nevents = nevents + T;
end
traces = cell(1,nevents);
event = 0;
for i=1:ncases
     T = size(dataTraining{i},1);
    for j=1:T
            event = event + 1;
            traces{event} = dataTraining{i}(1:j,2); 
            path = sprintf('%.0f,' , dataTraining{i}(1:j,2));
            if (isKey(td,path))
                td(path) = td(path) +1;
            else
                td(path) = 1;
            end
    end
end

ncases = size(dataTesting,1);
pred = zeros(1,ncases);
real_value = cell(1,ncases);
for i=1:ncases
    T = size(dataTesting{i},1);
    keyfound = true;
    mf = zeros(1,unique_values{2});
    for j=1:T-1
        if keyfound
            for c=1:unique_values{2}
                path = sprintf('%.0f,' , dataTesting{i}(j:T-1,2));
                path = [path  num2str(c)  ','];
                if isKey(td,path) 
                    mf(c) = td(path);
                    keyfound = false;
                end    
            end
        end
    [M I] = max(mf);
    pred(i) = I;
    real_value{i} = dataTesting{i}(T,2);
    end
end
end


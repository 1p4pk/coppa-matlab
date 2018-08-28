function score = sensitivity(prediction, real_value)
%ACCURACY 
%   Calculates prediction sensitivity
%   TPR = TP/P
%   As we have a multiclass problem we will assume each activity once as
%   positive and calculate the weighted average in the end
%
% Input
%%      prediction = predicted values
%%      real_value = actual values
%
% Output
%%      score = accuracy of prediction

assert(isequal(size(prediction),size(real_value)),'Length of prediction vector different from vector with real values');
ncases = length(prediction);

numberOfSymbols = max([real_value{1,:}]);
sensitivity = zeros(1,numberOfSymbols);
for j=1:numberOfSymbols
    tp = 0;
    p = 0;
    for i=1:ncases
        if real_value{i} == j
            p = p + 1;
            if real_value{i} == prediction{i}
                tp = tp + 1;
            end
        end
    end
    if p>0
        sensitivity(j) = tp/p;
    else
        sensitivity(j) = NaN;
    end
end
score = mean(sensitivity,'omitnan');
if isnan(score)
    score = 0;
end
end


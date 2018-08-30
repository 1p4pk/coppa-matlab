function score = specificity(prediction, real_value)
%ACCURACY 
%   Calculates prediction specificity
%   TNR = TN/N
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
specificity = zeros(1,numberOfSymbols);

for j=1:numberOfSymbols
    tn = 0;
    n = 0;
    for i=1:ncases
        if real_value{i} ~= j
            n = n + 1;
            if real_value{i} == prediction{i}
                tn = tn + 1;
            end
        end
    end
    if n>0
        specificity(j) = tn/n;
    else
        specificity(j) = NaN;
    end
end
score = mean(specificity,'omitnan');
if isnan(score)
    score = 0;
end
end


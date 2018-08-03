function [translated_value] = map_to_name(number_attr, numeric_value)
%MAP_TO_NAME Summary of this function goes here
%   Detailed explanation goes here
    load('mapping_table.mat');
    translated_value = mapping{number_attr}{numeric_value};
end


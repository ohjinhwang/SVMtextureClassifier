function [training, training_data, training_label, test, test_data, test_label] = SvmTrainTest(numTraining)  

% clear all;

d = dir;

g = size(d,1)-2;
type = zeros(g,1);

numClass = 2;

for n = 3:length(d)
    
    if d(n).name(1) == 'H'
        
        type(n-2,1) = 1;
        
    else if d(n).name(1) == 'M'
            
            type(n-2,1) = 2;
            
%         else if d(n).name(2) == '3'
%                 
%                 type(n-2,1) = 3;
%                 
%             else if d(n).name(2) == '4'
%                     
%                     type(n-2,1) = 4;
%                     
%                 end
%                 
%             end
%             
%             
        end
        
    end
    
end

for m = 1:numClass
    
    [r c] = find(type(:,1) == m);
    %label10 = randperm(size(r,1)/numSlice);
    label{m,1} = randperm(size(r,1))';
    for n = 1:size(r)
        
        type(r(n),2) = label{m,1}(n,1);
        
    end
    
end
clear r c; 
% [r c] = find(type(:,1) == 2);
% label2 = randperm(size(r,1))';
% for n = 1:size(r)
%    
%     type(r(n),2) = label2(n,1);
%     
% end
% clear r c;
m = 1; vec = []; 
training_label = [];

% training set
while m <= numClass
    
    [r c] = find(type(:,1) == m & type(:,2) <= numTraining);
    z = load(d(r(1)+2).name);
    training_data{1,m} = d(r(1)+2).name; 
    v = z.vec;
    %v = v';
    vec = cat(1,vec,v);
    l = m;
    training_label = cat(1,training_label,l);
    for n = 2:size(r)
        z = load(d(r(n)+2).name);
        training_data{n,m} = d(r(n)+2).name;
        vec2 = z.vec;
        [rr cc] = find(vec2 == -1);
        if (size(rr,1) > 0)
            for a = 1:size(cc,2)
                vec2(1,cc(a)) = 0;
            end
        end
        clear rr cc;
        %vec2 = vec2';
        vec = cat(1,vec,vec2);
        l2 = m;
        training_label = cat(1,training_label,l2);
    end
    
    m = m + 1;
    
end

Min = min(min(vec));

vec3 = vec - Min; 
vec3 = vec3 * 1000;
max_sig = max(max(vec3));
min_sig = min(min(vec3));
mid_sig = 0.5*(max_sig - min_sig) + min_sig;
training = (vec3 - mid_sig)./mid_sig; % final feature matrix for training

clear vec; 
vec = []; m = 1; test_label = [];

%training_data2 = cat(1,training_data(:,1),training_data(:,2));

%load('D:\2018\01_MMtexture\04_results\20181120\test.mat');

% test set
while m <= numClass
    
    [r c] = find(type(:,1) == m & type(:,2) > numTraining); %size(label{m,1},1)-28);
    z = load(d(r(1)+2).name);
    test_data{1,m} = d(r(1)+2).name;
    v = z.vec;
    %v = v';
    vec = cat(1,vec,v);
    l = m;
    test_label = cat(1,test_label,l);
    for n = 2:size(r)
        z = load(d(r(n)+2).name);
        test_data{n,m} = d(r(n)+2).name;
        vec2 = z.vec;
        %vec2 = vec2';
        [rr cc] = find(vec2 == -1);
        if (size(rr,1) > 0)
            for a = 1:size(cc,2)
                vec2(1,cc(a)) = 0;
            end
            clear rr cc;
        end
        vec = cat(1,vec,vec2);
        l2 = m;
        test_label = cat(1,test_label,l2);
    end
    
    m = m + 1;
    
end

Min = min(min(vec));

vec3 = vec - Min;
vec3 = vec3 * 1000;
max_sig = max(max(vec3));
min_sig = min(min(vec3));
mid_sig = 0.5*(max_sig - min_sig) + min_sig;
test = (vec3 - mid_sig)./mid_sig; % final feature matrix for training

clear vec;
clear c d l l2 label label1 label2 labelall m n r type v vec2 z g numClass numTraining a max_sig med mid_sig Min min_sig scale vec3;


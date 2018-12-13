%% SVM texture classifier: a source code for data preparation, SVM model construction and result display 

%% Part 1: open a DICOM file and normalize signals using disks 

% open DICOM 

clear all;

pwd;
[filepath,name,ext] = fileparts(pwd);
numModal = 1;

for nm = 1:numModal 
    
   OpenDicom;
   img{nm,1} = ans.img;
    
end

t1 = img{1,1};

label = zeros(size(t1,1),size(t1,2),size(t1,3));

figure(1); imagesc(t1(:,:,ceil(size(t1,3)/2)),[0 1000]); colormap gray; 
hold on;

% normalization using disk 

for numpt = 1:2 

    [x y] = getpts;

    xx = floor(x);
    yy = floor(y);

    x_edge = [xx-3,xx+3,xx-3,xx+3];
    y_edge = [yy-3,yy-3,yy+3,yy+3];

    grid_x = [min(x_edge):max(x_edge)];
    grid_y = [min(y_edge):max(y_edge)];

    [ROW COL] = meshgrid(grid_x,grid_y);

    for r = 1:size(COL,1)
        for c = 1:size(COL,2)
        
            sig{1,1}(r,c,numpt) = img{1,1}(COL(r,c),ROW(r,c),11);
        
        end
    end

    disc_sig(numpt) = mean(mean(sig{1,1}(:,:,numpt)));

end 

norm = img{1,1} - mean(disc_sig);

%% Part 2: Marrow segmentation using a GrowCut algorithm 

figure(1); imagesc(norm(:,:,ceil(size(norm,3)/2)),[0 1000]); colormap gray;
hold on;

[x y] = getpts;

xx = floor(x);
yy = floor(y);

for n = (ceil(size(norm,3)/2))-4:(ceil(size(norm,3)/2))+4
    
    for a = 1:size(xx,1)
        
        label(yy(a),xx(a),n) = 1;
        
    end
    
end

[v w] = getpts; 

vv = floor(v);
ww = floor(w);

for n = (ceil(size(norm,3)/2))-4:(ceil(size(norm,3)/2))+4
    
    for a = 1:size(vv,1)
        
        label(ww(a),vv(a),n) = -1;
        
    end
    
end

labels_3D = zeros(size(norm,1),size(norm,2),size(norm,3));
mask = zeros(size(norm,1),size(norm,2),size(norm,3));


for n = (ceil(size(norm,3)/2))-4:(ceil(size(norm,3)/2))+4
    
    [labels_out, strengths] = growcut(norm(:,:,n),label(:,:,n));
    labels_out = medfilt2(labels_out,[3,3]);
    labels_3D(:,:,n) = labels_out;
    
end

mask = logical(labels_3D);
norm_mask = maskWithBinary(mask,norm);

%% Part 3: Feature extraction 

seg = norm_mask;
mid = ceil(size(seg,3)/2);

mkdir('window');

for m = 1:6 % 5 windows per slice (L1 to S1)
    
    label = zeros(size(seg,1),size(seg,2),size(seg,3));
    
    figure(11); imagesc(seg(:,:,mid)); % axis off;
    
    colormap jet; title(num2str(mid));
    hold on;
    
    [x y] = getpts;
    
    xx = floor(x);
    yy = floor(y);
    
    x_edge = [xx-19,xx+19,xx-19,xx+19];
    y_edge = [yy-13,yy-13,yy+13,yy+13];
    
    grid_x = [min(x_edge):max(x_edge)];
    grid_y = [min(y_edge):max(y_edge)];
    
    [ROW COL] = meshgrid(grid_x,grid_y);
    
    for num = 1:1
        
        for n = mid-4:mid+4
            
            o = mid-4;
            
            for r = 1:size(COL,1)
                for c = 1:size(COL,2)
                    
                    sig(r,c,n-o+1) = seg(COL(r,c),ROW(r,c),n);
                    
                end
            end

            vec = reshape(sig(:,:,n-o+1),[1,size(sig(:,:,n-o+1),2)*size(sig(:,:,n-o+1),1)]);
            address = pwd;
            [pathstr,filename,ext] = fileparts(address);
            name0 = [filename,'_' num2str(num)];
            name1 = ['_s0' num2str(n-o+1)]; % 9 slices in total
            name2 = ['_w0' num2str(m),'.mat'];
            name = [name0 name1 name2];
            cd('window');
            save(name,'vec');
            cd ..;
            
        end
        
    end

end

cd('window');
concat;

%% Part 4: Data preparation for SVM 

function [training, training_data, training_label, test, test_data, test_label] = SvmTrainTest(numTraining)  

d = dir;

g = size(d,1)-2;
type = zeros(g,1);

numClass = 2;

for n = 3:length(d)
    
    if d(n).name(1) == 'H'
        
        type(n-2,1) = 1;
        
    else if d(n).name(1) == 'M'
            
            type(n-2,1) = 2;
            
        end
        
    end
    
end

for m = 1:numClass
    
    [r c] = find(type(:,1) == m);
    label{m,1} = randperm(size(r,1))';
    for n = 1:size(r)
        
        type(r(n),2) = label{m,1}(n,1);
        
    end
    
end
clear r c; 
m = 1; vec = []; 
training_label = [];

% training set
while m <= numClass
    
    [r c] = find(type(:,1) == m & type(:,2) <= numTraining);
    z = load(d(r(1)+2).name);
    training_data{1,m} = d(r(1)+2).name; 
    v = z.vec;
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

% test set
while m <= numClass
    
    [r c] = find(type(:,1) == m & type(:,2) > numTraining); %size(label{m,1},1)-28);
    z = load(d(r(1)+2).name);
    test_data{1,m} = d(r(1)+2).name;
    v = z.vec;
    vec = cat(1,vec,v);
    l = m;
    test_label = cat(1,test_label,l);
    for n = 2:size(r)
        z = load(d(r(n)+2).name);
        test_data{n,m} = d(r(n)+2).name;
        vec2 = z.vec;
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

%% Part 5: Two class SVM texture classifier for separate training and test sets 

for n = 1:1
    
    numTraining = 180; % per class 
    [training, training_data, training_label, test, test_data, test_label] = SvmTrainTest(numTraining);
    [CA_test, CA_training decisValueWinner] = demo_libsvm_test12(1, training, training_label, test, test_label); 
    CATEST(n,:) = CA_test;
    CATRAINING(n,:) = CA_training; 
    decisValue(:,n) = decisValueWinner(:,1); 
    x(:,1) = decisValueWinner(:,1); 
    x(:,2) = test_label-1; 
    x(:,3) = abs(x(:,1)-x(:,2));
    x(:,4) = round(x(:,3));

    rocout = roc(x); 
    auc(n,1) = rocout.AUC; 
    
end % training/test separated 


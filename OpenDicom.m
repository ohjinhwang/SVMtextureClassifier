function image = OpenDicom
% browse for .dcm folder
dd = dir;

%imgdir = [pwd '\' dd(num+2).name];


imgdir = pwd;
% Add final '\' to path name if missing
if ~isequal(imgdir(length(imgdir)),filesep)
    imgdir = [imgdir filesep];
end

% Get the directory
d = dir(imgdir);

% number of files
numslice = length(d) - 2; % avoid . and ..

% generate b matrix (number of b) 
bb = 0;
i = 1;
for n=1:numslice
    filename=char(d(n+2,1).name);
    try 
        info{n,1} = dicominfo([imgdir,filename]);
        if (isfield(info{n,1},'Private_0019_100c') == 1)
            b = info{n,1}.Private_0019_100c;
            bmat(i) = b;
            i = i+1;
        end
        image.echotime(n,1) = info{n,1}.EchoTime;
    catch
        % errordlg('Folder contains no DICOM files','File Error');
        break;
    end
end

% image reconstruction

% if the DICOM file contains Diffusion data:

% if (isfield(info{n,1},'Private_0019_100c') == 1)
%     [r c] = find(bmat == 0);
%     numb = c(1,2) - c(1,1);
%     numscan = numslice/numb;
%     image.numslice = numscan;
% end

image.row = info{1}.Rows;
image.col = info{1}.Columns;
image.totalslice = numslice;
%image.voxelX = info{1}.PixelSpacing(1); 
%image.voxelY = info{1}.PixelSpacing(2); 
%image.voxelZ = info{1}.SliceThickness;
%image.imgdir = imgdir; 
image.info = info;

if (isfield(info{1},'RescaleSlope') == 1)
    image.RescaleSlope = info{1}.RescaleSlope;
end
if (isfield(info{1},'RescaleIntercept') == 1)
    image.RescaleIntercept = info{1}.RescaleIntercept;
end
row = image.row;
col = image.col;
image.img = zeros(row,col,numslice);

%if (isfield(info{n,1},'Private_0019_100c') == 1) % for diffusion
    %     for nn=1:numb
    %         for m=1:numslice
    %             ima = dicomread(info{n,1});
    %             if info{n,1}.Private_0019_100c == bmat(nn)
    %                 numscan = info{n,1}.Private_0019_100a;
    %                 image.img(:,:,nn,numscan) = ima;
    %             end
    %         end
    %     end
    %     image.numb = numb;
    %     i = 1;
    %     for n = 1:numb
    %         b(i) = bmat(1,i);
    %         i = i+1;
    %     end
    %     image.b = b;
    % else % for any image except for diffusion
    for m = 1:numslice
        filename = char(d(m+2,1).name);
        info = dicominfo([imgdir,filename]);
        ima = dicomread(info);
        ima = double(ima);
        if (isfield(image,'RescaleSlope') == 1)
            
            ima = image.RescaleSlope * ima + image.RescaleIntercept;
            
        end
        if (isfield(image,'RescaleIntercept') == 1)
            
            if (min(min(ima)) < -pi && max(max(ima)) > pi)
                
                ima = - (ima / image.RescaleIntercept) * pi;
                
            end
            
        end
        
        %ima = double(ima);
        image.img(:,:,m) = ima;
    end
%end

z = 1;
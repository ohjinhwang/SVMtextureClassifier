function signal = maskWithBinary(binary,img)

% written by Eo-Jin Hwang, 02/22/14 (ohjinhwang@gmail.com)
% accepts 3D matrix of img and a binary matrix 
% create masked img using binary data

[row col numslice] = size(img);
signal = zeros(row,col,numslice);

for n = 1:numslice
    for r = 1:row
        for c = 1:col
           
            if (binary(r,c,n) > 0)
                
                signal(r,c,n) = img(r,c,n);
                
            end
        end
    end
end

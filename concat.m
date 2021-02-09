clear all;

d = dir;

% single slice concatenation

for n = 3:length(d)
    
    window(n-2,1) = str2num(d(n).name(end-4));
    slice(n-2,1) = str2num(d(n).name(end-8));
    modality(n-2,1) = str2num(d(n).name(end-12));
    
end

numMod = size(unique(modality),1);
numSlice = size(unique(slice),1);
numWindow = size(unique(window),1);

v = load(d(3).name);
vec = v.vec;
vecs = [];

mkdir('window_slice');

for n = 3:length(d)-1
    
    m = modality(n-2,1);
    s = slice(n-2,1);
    
    if (slice(n-2+1,1) ~= s) % save window vector in folder 'window_slice'
        
        [pathstr name ext] = fileparts(pwd);
        [pathstr name ext] = fileparts(pathstr);
        filename = [name '_' num2str(m) '_s0' num2str(s) '.mat'];
        cd('window_slice');
        save(filename,'vec');
        cd ..;
        v = load(d(n).name);
        vec = v.vec;
        
    else if n == length(d)-1
            
            v = load(d(n+1).name);
            vec2 = v.vec;
            vec = cat(2,vec,vec2);
            [pathstr name ext] = fileparts(pwd);
            [pathstr name ext] = fileparts(pathstr);
            filename = [name '_' num2str(m) '_s0' num2str(s) '.mat'];
            cd('window_slice');
            save(filename,'vec');
            cd ..;

        else
            
            v = load(d(n+1).name);
            vec2 = v.vec;
            vec = cat(2,vec,vec2);
            
        end
        
    end
    
end

cd('window_slice');

e = dir;

for n = 3:length(e)
    
   modal(n-2,1) = str2num(e(n).name(end-8));
    
end

v = load(e(3).name);
vec = v.vec;

mkdir('window_sliceAll');

for n = 3:length(e)-1
    
    m = modal(n-2,1);
    
    if (modal(n-2+1,1) ~= m)
        
        cd ..;
        [pathstr name ext] = fileparts(pwd);
        [pathstr name ext] = fileparts(pathstr);
        filename = [name '_' num2str(m) '_s.mat'];
        cd('window_slice');
        cd('window_sliceAll');
        save(filename,'vec');
        cd ..;
        v = load(e(n).name);
        vec = v.vec;
        
    else if n == length(e)-1
            
            v = load(e(n+1).name);
            vec2 = v.vec;
            vec = cat(2,vec,vec2);
            cd ..;
            [pathstr name ext] = fileparts(pwd);
            [pathstr name ext] = fileparts(pathstr);
            filename = [name '_' num2str(m) '_s.mat'];
            cd('window_slice');
            cd('window_sliceAll');
            save(filename,'vec');
            cd ..;
            
        else
            
            v = load(e(n+1).name);
            vec2 = v.vec;
            vec = cat(2,vec,vec2);
            
        end
        
    end
    
end

% cd ..;
% 
% mkdir('window_marrow');
% dd = dir;
% dd = dd(3:end-2); 
% 
% for nm = 1:numMod
%     
%     ddd = dd(((numSlice*numWindow)*(nm-1)+1):numSlice*numWindow*nm); 
%     
%     for nw = 1:numWindow
%         
%         v = load(ddd(nw).name);
%         vec = v.vec;
%         
%         for n = 1+numWindow:length(ddd)
%             
%             w = window(n,1);
%             
%             if (w == nw)
%                 
%                 v = load(ddd(n).name);
%                 vec2 = v.vec;
%                 vec = cat(2,vec,vec2);
%                 
%             end
%             
%         end
%         
%         [pathstr name ext] = fileparts(pwd);
%         [pathstr name ext] = fileparts(pathstr);
%         filename = [name '_' num2str(nm) '_w0' num2str(nw) '.mat'];
%         cd('window_marrow');
%         save(filename,'vec');
%         cd ..;
%         
%     end
% 
% end
% 
% cd('window_marrow');
% 
% ee = dir;
% for n = 3:length(ee)
%     
%    modal(n-2,1) = str2num(ee(n).name(end-8));
%     
% end
% 
% mkdir('window_marrowAll');
% 
% for n = 3:length(ee)-2
%     
%     m = modal(n-2,1);
%     
%     if (modal(n-2+1,1) ~= m)
%         
%         cd ..;
%         [pathstr name ext] = fileparts(pwd);
%         [pathstr name ext] = fileparts(pathstr);
%         filename = [name '_' num2str(m) '_m.mat'];
%         cd('window_marrow');
%         cd('window_marrowAll');
%         save(filename,'vec');
%         cd ..;
%         v = load(ee(n).name);
%         vec = v.vec;
%         
%     else if n == length(ee)-2
%             
%             v = load(ee(n+1).name);
%             vec2 = v.vec;
%             vec = cat(2,vec,vec2);
%             cd ..;
%             [pathstr name ext] = fileparts(pwd);
%             [pathstr name ext] = fileparts(pathstr);
%             filename = [name '_' num2str(m) '_m.mat'];
%             cd('window_marrow');
%             cd('window_marrowAll');
%             save(filename,'vec');
%             cd ..;
%             
%         else
%             
%             v = load(ee(n+1).name);
%             vec2 = v.vec;
%             vec = cat(2,vec,vec2);
%             
%         end
%         
%     end
%     
% end
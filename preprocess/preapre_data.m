
clear
% [x1,y1,c1,x2,y2,c2,x3,y3,c3]
% val.bodies(1).joints 

video_name = 1:10;
for name_index = video_name
    files = dir(['E:\wifiposedata\train80\oct17set', num2str(name_index),'_*.mat']); 
    
    for i = 1:length(files) 
        if  isempty(strfind(files(i).name, 'two'))&& isempty(strfind(files(i).name, 'three'))&& ...
            isempty(strfind(files(i).name, 'four'))&& isempty(strfind(files(i).name, 'five'))
            
            frame_index = getIndex(files(i).name);
            % load coco-18
            fname = ['set', num2str(name_index), '\sep-json\', num2str(frame_index-1), '.json'];
            if ~isempty(dir(fname))
                
            fid = fopen(fname); 
            raw = fread(fid,inf); 
            str = char(raw'); 
            fclose(fid); 
            val = jsondecode(str);
            joints = val.bodies(1).joints;
            
            x = joints(1:3:end);
            y = joints(2:3:end);
            c = joints(3:3:end);
            
            
            jointsVector = [x;y;c;c];
            
            jointsMatrix = zeros([18,18,4]);
            
            for row = 1:18
                for column = 1:18
                    if row == column
                        jointsMatrix(row,column,:) = [x(row),y(row),c(row),c(row)];
                    else
                        jointsMatrix(row,column,:) = [x(row)-x(column),y(row)-y(column),c(row)*c(column),c(row)*c(column)];
                    end 
                end
            end
            files(i).name
            load([files(1).folder, '\', files(i).name], 'csi_serial', 'frame');
            save(['train80singleperson\', files(i).name], 'csi_serial', 'frame', 'jointsVector', 'jointsMatrix', '-v7.3')
            end
        end  
        % save([files(1).folder, '\', files(i).name], 'heatmapscoco18', '-append');
    end
 
end


function index = getIndex(file_name)
    for num_length = [5,4,3,2,1]
        file_index = file_name(end-4-num_length:end-4);
        if ~isempty(str2num(file_index))
            index = str2num(file_index);
            break;
        end

    end

end

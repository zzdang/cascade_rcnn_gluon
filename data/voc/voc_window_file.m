% Cascade-RCNN
% Copyright (c) 2018 The Regents of the University of California
% see cascade-rcnn/LICENSE for details
% Written by Zhaowei Cai [zwcai-at-ucsd.edu]
% Please email me if you find bugs, or have suggestions or questions!

clear all; close all;

% setup VOC directory
rootDir = '/your/VOC/path'; % your VOC directory has dataDir/VOCdevkit
dev = [rootDir '/VOCdevkit']; addpath(genpath([dev '/VOCcode']));
VOCinit; C = VOCopts.classes'; catsMap = containers.Map(C,1:length(C));
%years = {'2007'}; split = 'test';
years = {'2007'; '2012'}; split = 'trainval';
nData = numel(years);

% writing window files
yearStr = ''; for k = 1:nData, yearStr=[yearStr years{k}(3:4)]; end
fileName = sprintf('window_file_voc%s_%s.txt',yearStr,split);
fid = fopen(fileName, 'wt');

show = 0;
if (show)
  fig = figure(1); set(fig,'Position',[-30 30 600 600]);
  h.axes = axes('position',[0.1,0.1,0.8,0.8]);
end

% running
idCount = 0;
for k = 1:nData
  fprintf('VOC, year: %s, split: %s\n', years{k}, split);
  f = fopen([dev '/VOC' years{k} '/ImageSets/Main/' split '.txt']);
  idList = textscan(f,'%s %*s'); idList=idList{1}; fclose(f); n=length(idList);
  for i=1:n
    if (mod(i,1000) == 0), fprintf('idx: %i\n', i); end
    nm=[idList{i} '.jpg'];
    imgFullName = sprintf('VOC%s/JPEGImages/%s',years{k},nm);
    imgPath = sprintf('%s/%s',dev,imgFullName);
    f=[dev '/VOC' years{k} '/Annotations/' idList{i} '.xml'];
    R=PASreadrecord(f);  object=R.objects;
    t=catsMap.values({object.class}); clsIds=[t{:}];
    
    if (show), I = imread(imgPath);
      cla(h.axes); imshow(I); axis(h.axes,'image','off');
    end
    
    nObjs = numel(object); chw=R.imgsize([3 2 1]);
    fprintf(fid, '# %d\n', idCount);
    fprintf(fid, '%s\n', imgFullName);
    fprintf(fid, '%d\n%d\n%d\n', chw(1), chw(2), chw(3));
    fprintf(fid, '%d\n', nObjs);
    
    for j = 1:nObjs
      obj = object(j); bbox = obj.bbox;
      if (bbox(3)<=bbox(1) || bbox(4)<=bbox(2))
        continue;
      end    
      bbox=bbox-1;
      fprintf(fid, '%i %i %i %i %i %i %i\n', clsIds(j), 0, obj.difficult,...
          bbox(1), bbox(2), bbox(3), bbox(4));
      if (show)
        wh=bbox(3:4)-bbox(1:2)+1;
        if (obj.difficult), color = 'red';
        else color = 'yellow'; end
        rectangle('Position', [bbox(1) bbox(2) wh],'LineWidth',2,'edgecolor',color);   
        text(bbox(1)+0.5*wh(1),bbox(2),obj.class,'color','r','BackgroundColor','k',...
            'HorizontalAlignment','center','VerticalAlignment','bottom','FontWeight',...
            'bold','FontSize',8);
      end
     end
     fprintf(fid, '%d\n', 0);
     idCount = idCount+1;
  end
end
fclose(fid);

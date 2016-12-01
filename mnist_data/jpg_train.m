%%
clc;
clear all;

%images = loadMNISTImages('t10k-images-idx3-ubyte');
%labels = loadMNISTLabels('t10k-labels-idx1-ubyte');

images = loadMNISTImages('train-images-idx3-ubyte');
labels = loadMNISTLabels('train-labels-idx1-ubyte');
save('train_label.mat','labels')

[w h]=size(images)
endIndex = h;

dindex=zeros(10,1);

% mkdir('./data2/') 
% for j=1:endIndex
%     aImg = reshape( images(:,j), 28, 28);
%     imshow(aImg);
%     
%     label = labels(j);
%     dindex( label+1 ) = dindex( label+1 ) +1;
%     dataD = strcat('./data2/mnist_', num2str( label ), '_', num2str( dindex(label+1) ), '.jpg' );    
%     %dindex
%     imwrite(aImg, dataD);
%     dataD;
% end

% mkdir('./train/') 
% for j=1:endIndex
%     aImg = reshape( images(:,j), 28, 28);
%     %imshow(aImg);
%     
%     label = labels(j);
%     dindex( label+1 ) = dindex( label+1 ) +1;
%     dataD = strcat('./train/mnist_', num2str(j), '.jpg' );    
%     %dindex
%     imwrite(aImg, dataD);
%     dataD;
% end

fileID = fopen('train.txt','w');
%make path and label
dindex=zeros(10,1);
for j=1:endIndex
    label = labels(j);
    dindex( label+1 ) = dindex( label+1 ) +1;
    dataD = strcat('/train/mnist_', num2str(j), '.jpg' );
    fprintf(fileID,'%s %d\n',dataD, label);    
end
fclose(fileID);


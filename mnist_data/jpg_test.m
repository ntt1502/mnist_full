%%
clc;
clear all;

images = loadMNISTImages('t10k-images-idx3-ubyte');
labels = loadMNISTLabels('t10k-labels-idx1-ubyte');
save('test_label.mat','labels')
%images = loadMNISTImages('train-images-idx3-ubyte');
%labels = loadMNISTLabels('train-labels-idx1-ubyte');

[w h]=size(images)
endIndex = h;

dindex=zeros(10,1);

% Dat ten voi STT trong moi label
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

mkdir('./test/') 
for j=1:endIndex
    aImg = reshape( images(:,j), 28, 28);
    %imshow(aImg);
    
    label = labels(j);
    dindex( label+1 ) = dindex( label+1 ) +1;
    dataD = strcat('./test/mnist_', num2str(j), '.jpg' );    
    %dindex
    imwrite(aImg, dataD);
    dataD;
end

% Dat ten voi STT trong moi label
% fileID = fopen('test.txt','w');
% %make path and label
% dindex=zeros(10,1);
% for j=1:endIndex
%     
%     label = labels(j);
%     dindex( label+1 ) = dindex( label+1 ) +1;
%     dataD = strcat('/test/mnist_', num2str( label ), '_', num2str( dindex(label+1) ), '.jpg' );
%     fprintf(fileID,'%s %d\n',dataD, label);    
% end

fileID = fopen('test.txt','w');
%make path and label
dindex=zeros(10,1);
for j=1:endIndex
    
    label = labels(j);
    dindex( label+1 ) = dindex( label+1 ) +1;
    dataD = strcat('/test/mnist_', num2str(j), '.jpg' );
    fprintf(fileID,'%s %d\n',dataD, label);    
end
fclose(fileID);

